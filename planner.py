"""
Simple OMPL Motion Planner - Uses visible robot for collision checking

Interface similar to pybullet_ompl with visible collision checking for easy debugging.

New in this version
-------------------
* ``sample_ik_candidates`` – generate IK candidates for exactly **two yaw
  variants** (grasp aligned with cube X axis / cube Y axis, i.e. 90° apart
  about world Z).  Tries multiple seeds per variant, collision-checks each,
  and returns the top-N ranked by distance-to-start + joint-limit cost.

* ``score_ik_candidate`` – standalone scoring function (distance-to-start +
  joint-limit cost).

* ``RobotOMPLPlanner.plan()`` now accepts **a list of goal configs** (produced
  by ``sample_ik_candidates``).  Goals are tried in order; the first
  successful plan is returned.  Each attempt uses a two-stage fallback:
  RRTConnect (fast feasibility) → configured planner.

* ``RobotOMPLPlanner.plan_to_pose()`` – one-shot API: given a target EEF pose
  and the cube's base orientation, generates IK candidates for the X/Y yaw
  variants and plans to the best one.

Backward compatibility
----------------------
Existing code calling ``plan(start, goal_config)`` with a flat list still
works unchanged.

Usage (new API)
---------------
    from planner import RobotOMPLPlanner, sample_ik_candidates

    planner = RobotOMPLPlanner(robot, obstacles)
    planner.set_planner("AITstar")

    # Option A: high-level one-shot API
    success, path = planner.plan_to_pose(
        start_config=current_joints,
        pos=target_pos,
        base_orn=target_orn,   # two yaw variants auto-generated
        planning_time=10.0,
    )

    # Option B: manual IK sampling + multi-goal planning
    candidates = sample_ik_candidates(robot, planner, target_pos, target_orn,
                                       q_start=current_joints)
    success, path = planner.plan(current_joints, candidates)

    # Option C: original single-goal API (unchanged)
    success, path = planner.plan(start_joints, goal_joints)

    if success:
        planner.execute(path)

Usage (original)
----------------
    from planner import RobotOMPLPlanner, FrankaRobot, solve_ik_collision_free

    robot = FrankaRobot([0, 0, 0.63], [0, 0, 0])
    robot.load()

    planner = RobotOMPLPlanner(robot, obstacles)
    planner.set_planner("AITstar")

    start = solve_ik_collision_free(robot, planner, start_pos, start_orn)
    goal  = solve_ik_collision_free(robot, planner, goal_pos,  goal_orn)

    res, path = planner.plan(start, goal, planning_time=10.0)

    if res:
        planner.execute(path)
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
from collections import namedtuple

import ompl.base as ob
import ompl.geometric as og


def _quat_mul(q1, q2):
    """
    Hamilton product of two quaternions in PyBullet convention [x, y, z, w].
    q1 is applied *after* q2 (i.e. result = q1 * q2).
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def _two_yaw_variants(base_orn):
    """
    Given a base end-effector orientation, return exactly two quaternions that
    differ by 90° about the **world Z axis**.

    These correspond to grasping a cube aligned with its X axis (``base_orn``)
    and grasping aligned with its Y axis (``base_orn`` rotated 90° about world
    Z).  Both are geometrically valid symmetric grasps; trying both dramatically
    increases IK / planning success.

    Convention
    ----------
    - Rotation axis: world +Z  (equivalent to tool Z for top-down grasps).
    - Rotation angle: 0° and +90°.

    Args:
        base_orn: quaternion [x, y, z, w]  (PyBullet convention)

    Returns:
        list of two quaternions [[x,y,z,w], [x,y,z,w]]
    """
    base = np.array(base_orn, dtype=float)
    # 90° rotation about world +Z: q_rot = [0, 0, sin(π/4), cos(π/4)]
    s = np.sin(np.pi / 4.0)
    c = np.cos(np.pi / 4.0)
    q_rot_z90 = np.array([0.0, 0.0, s, c])
    rotated = _quat_mul(q_rot_z90, base)
    return [base.tolist(), rotated.tolist()]


def score_ik_candidate(q, q_start, lower, upper):
    """
    Score an IK candidate – **lower is better**.

    score = ||q – q_start||₂  +  0.5 * joint_limit_cost(q)

    The joint-limit cost penalises configurations that are close to their
    joint limits:

        joint_limit_cost = Σ  |q_i – midpoint_i| / half_range_i

    so every joint contributes a number in [0, 1] where 0 is at the midpoint
    and 1 is at the limit.

    Args:
        q:       candidate config, array-like of length n
        q_start: current/start config, array-like of length n
        lower:   lower joint limits, array-like of length n
        upper:   upper joint limits, array-like of length n

    Returns:
        float – combined cost (distance-to-start + limit penalty)
    """
    q       = np.array(q,       dtype=float)
    q_start = np.array(q_start, dtype=float)
    lower   = np.array(lower,   dtype=float)
    upper   = np.array(upper,   dtype=float)

    dist_to_start = np.linalg.norm(q - q_start)

    midpoint   = (lower + upper) / 2.0
    half_range = (upper - lower) / 2.0
    # Avoid division by zero for fixed joints (range == 0)
    half_range = np.where(half_range > 1e-6, half_range, 1.0)
    limit_cost = np.sum(np.abs(q - midpoint) / half_range)

    return float(dist_to_start + 0.5 * limit_cost)


def sample_ik_candidates(robot, planner, pos, base_orn,
                          q_start=None,
                          num_seeds_per_yaw=15,
                          pos_tol=0.01,
                          orn_tol=0.05,
                          top_n=5,
                          dedup_tol=0.05):
    """
    Generate and rank IK candidates for a cube pick grasp.

    Tries **exactly two yaw variants** (0° and 90° about world Z relative to
    ``base_orn``) with ``num_seeds_per_yaw`` random IK seeds each.  Each
    candidate is checked for:

    1. Joint bound satisfaction
    2. FK accuracy (position + orientation within tolerances)
    3. Collision freedom (via ``planner.is_state_valid_list``)

    Candidates are then scored with :func:`score_ik_candidate` and the best
    ``top_n`` are returned.

    Args:
        robot:             Robot object with arm_controllable_joints, arm_lower_limits,
                           arm_upper_limits, eef_id, id.
        planner:           ``RobotOMPLPlanner`` instance (used for collision checking).
        pos:               Target EEF position [x, y, z].
        base_orn:          Base orientation quaternion [x, y, z, w].  One of the
                           two yaw variants is this quaternion; the other is
                           rotated 90° about world Z.
        q_start:           Current joint configuration used for scoring.  If
                           ``None``, the robot's actual current state is used.
        num_seeds_per_yaw: Number of random IK seeds tried for each yaw variant
                           (default 15, so 30 attempts total).
        pos_tol:           Max allowed FK position error in metres (default 1 cm).
        orn_tol:           Max allowed FK orientation error in radians (default
                           ~2.9°).
        top_n:             Maximum number of candidates returned (default 5).
        dedup_tol:         Two candidates closer than this L2 distance (radians)
                           are treated as duplicates; only the better-scored one
                           is kept (default 0.05 rad).

    Returns:
        list of joint-config lists, sorted best-first (empty list if none found).

    Example::

        candidates = sample_ik_candidates(robot, planner, pos, base_orn,
                                          q_start=current_joints)
        success, path = planner.plan(start_joints, candidates)
    """
    lower    = np.array(robot.arm_lower_limits)
    upper    = np.array(robot.arm_upper_limits)
    n_joints = len(robot.arm_controllable_joints)

    if q_start is None:
        q_start = [p.getJointState(robot.id, j)[0]
                   for j in robot.arm_controllable_joints]
    q_start = np.array(q_start)

    yaw_variants = _two_yaw_variants(base_orn)
    rest_poses   = getattr(robot, 'arm_rest_poses',
                           [0.0] * n_joints)

    candidates = []  # list of (score, config)

    for yaw_idx, orn in enumerate(yaw_variants):
        yaw_label = ["cube-X axis", "cube-Y axis"][yaw_idx]
        found = 0
        for seed_idx in range(num_seeds_per_yaw):
            # First seed: use rest pose; subsequent: random
            if seed_idx == 0:
                seed = rest_poses
            else:
                seed = (lower + np.random.rand(n_joints) * (upper - lower)).tolist()

            for i, j in enumerate(robot.arm_controllable_joints):
                p.resetJointState(robot.id, j, seed[i])

            q_raw = p.calculateInverseKinematics(
                robot.id, robot.eef_id, pos, orn,
                lowerLimits=robot.arm_lower_limits,
                upperLimits=robot.arm_upper_limits,
                jointRanges=[upper[i] - lower[i] for i in range(n_joints)],
                restPoses=rest_poses,
                maxNumIterations=200,
                residualThreshold=1e-5,
            )
            q = list(q_raw[:n_joints])

            # 1. Bounds check
            q_arr = np.array(q)
            if np.any(q_arr < lower - 1e-4) or np.any(q_arr > upper + 1e-4):
                continue
            q = np.clip(q_arr, lower, upper).tolist()

            # 2. FK verification
            for i, j in enumerate(robot.arm_controllable_joints):
                p.resetJointState(robot.id, j, q[i])
            eef_state  = p.getLinkState(robot.id, robot.eef_id,
                                        computeForwardKinematics=True)
            actual_pos = np.array(eef_state[4])
            actual_orn = np.array(eef_state[5])

            pos_err = np.linalg.norm(actual_pos - np.array(pos))
            dot     = np.clip(abs(np.dot(actual_orn, np.array(orn))), 0.0, 1.0)
            orn_err = 2.0 * np.arccos(dot)

            if pos_err > pos_tol or orn_err > orn_tol:
                continue

            # 3. Collision check
            if not planner.is_state_valid_list(q):
                continue

            score = score_ik_candidate(q, q_start, lower, upper)
            candidates.append((score, q))
            found += 1

        print(f"  IK sampling ({yaw_label}): {found} collision-free candidates")

    if not candidates:
        print("  ✗ sample_ik_candidates: no valid IK found for either yaw variant")
        return []

    # Sort by score (best = lowest) and deduplicate near-identical configs
    candidates.sort(key=lambda x: x[0])
    ranked = []
    for _, q in candidates:
        q_arr = np.array(q)
        duplicate = any(np.linalg.norm(q_arr - np.array(r)) < dedup_tol for r in ranked)
        if not duplicate:
            ranked.append(q)
        if len(ranked) >= top_n:
            break

    print(f"  ✓ sample_ik_candidates: returning {len(ranked)} ranked candidates")
    return ranked


def visualise_eef_traj(current_eef_pos, prev_line_id=None):
    """
    Visualizes end-effector trajectory by drawing a small sphere at current position
    and a line to the previous position if available.
    
    Args:
        current_eef_pos: Current end-effector position [x, y, z]
        prev_line_id: Previous line ID to maintain trajectory
        
    Returns:
        line_id: ID of the drawn line segment
    """
    # Draw a small red sphere at current EEF position
    p.addUserDebugPoints(
        pointPositions=[current_eef_pos],
        pointColorsRGB=[[1, 0, 0]],  # Red color
        pointSize=5,
        lifeTime=0  # Permanent
    )
    
    # Store previous position for line drawing
    if not hasattr(visualise_eef_traj, 'prev_pos'):
        visualise_eef_traj.prev_pos = None
    
    line_id = None
    if visualise_eef_traj.prev_pos is not None:
        # Draw line from previous to current position
        line_id = p.addUserDebugLine(
            lineFromXYZ=visualise_eef_traj.prev_pos,
            lineToXYZ=current_eef_pos,
            lineColorRGB=[0, 1, 0],  # Green color
            lineWidth=2,
            lifeTime=0  # Permanent
        )
    
    visualise_eef_traj.prev_pos = current_eef_pos
    return line_id
def solve_ik(robot, planner, pos, orn):
    q = p.calculateInverseKinematics(
            robot.id, robot.eef_id, pos, orn,
            lowerLimits=robot.arm_lower_limits,
            upperLimits=robot.arm_upper_limits,
            jointRanges=[robot.arm_upper_limits[i] - robot.arm_lower_limits[i]
                         for i in range(len(robot.arm_controllable_joints))],
            restPoses=getattr(robot, 'arm_rest_poses', [0.0] * len(robot.arm_controllable_joints)),
            maxNumIterations=100,
            residualThreshold=1e-5
        )
    return q

def solve_ik_collision_free(robot, planner, pos, orn, max_attempts=50,
                             pos_tol=0.01, orn_tol=0.05):
    """
    Solve IK with collision checking, bounds validation, and FK verification.
    
    Args:
        pos_tol: Max allowed EEF position error in metres (default 1cm)
        orn_tol: Max allowed EEF orientation error in radians (default ~3deg)
    """
    lower = np.array(robot.arm_lower_limits)
    upper = np.array(robot.arm_upper_limits)
    
    for attempt in range(max_attempts):
        # Use random seed after first attempt
        if attempt > 0:
            seed = lower + np.random.rand(len(robot.arm_controllable_joints)) * (upper - lower)
            for i, j in enumerate(robot.arm_controllable_joints):
                p.resetJointState(robot.id, j, seed[i])
        
        # Compute IK
        q = p.calculateInverseKinematics(
            robot.id, robot.eef_id, pos, orn,
            lowerLimits=robot.arm_lower_limits,
            upperLimits=robot.arm_upper_limits,
            jointRanges=[robot.arm_upper_limits[i] - robot.arm_lower_limits[i]
                         for i in range(len(robot.arm_controllable_joints))],
            restPoses=getattr(robot, 'arm_rest_poses', [0.0] * len(robot.arm_controllable_joints)),
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        q = list(q[:len(robot.arm_controllable_joints)])
        
        # 1. Bounds check
        q_array = np.array(q)
        if np.any(q_array < lower - 1e-4) or np.any(q_array > upper + 1e-4):
            if attempt == 0:
                print(f"  ⚠ IK violates bounds, retrying...")
            continue
        q = np.clip(q_array, lower, upper).tolist()
        
        # 2. ── FK VERIFICATION ──────────────────────────────────────────────
        # Apply joints and read back actual EEF pose via forward kinematics.
        # PyBullet IK can silently fail (joint limits, singularities, etc.)
        # and return a solution that puts the EEF nowhere near the target.
        for i, j in enumerate(robot.arm_controllable_joints):
            p.resetJointState(robot.id, j, q[i])
        
        eef_state   = p.getLinkState(robot.id, robot.eef_id,computeForwardKinematics=True)
        actual_pos  = np.array(eef_state[4])   # world position  (link frame)
        actual_orn  = np.array(eef_state[5])   # world quaternion

        pos_error = np.linalg.norm(actual_pos - np.array(pos))

        # Quaternion angular distance: 2*arccos(|q1·q2|)  (handles q == -q)
        target_orn  = np.array(orn)
        dot         = np.clip(abs(np.dot(actual_orn, target_orn)), 0.0, 1.0)
        orn_error   = 2.0 * np.arccos(dot)

        if pos_error > pos_tol or orn_error > orn_tol:
            if attempt < 3:
                print(f"  ⚠ FK mismatch (attempt {attempt+1}): "
                      f"pos_err={pos_error*1000:.1f}mm  "
                      f"orn_err={np.degrees(orn_error):.1f}°  — retrying")
            continue
        # ── end FK check ───────────────────────────────────────────────────

        # 3. Collision check
        if planner.is_state_valid_list(q):
            print(f"  ✓ IK valid (attempt {attempt+1}): "
                  f"pos_err={pos_error*1000:.2f}mm  "
                  f"orn_err={np.degrees(orn_error):.2f}°")
            return q
        else:
            if attempt < 3:
                planner._debug_collision_state(q)
    
    print(f"  ✗ No valid IK after {max_attempts} attempts")
    return None
class RobotOMPLPlanner:
    """
    Simple OMPL motion planner using visible robot for collision checking.
    
    Works with any robot class (duck typing) that provides:
    - robot.id: PyBullet body ID
    - robot.arm_controllable_joints: list of joint indices
    - robot.arm_lower_limits: list of lower limits
    - robot.arm_upper_limits: list of upper limits
    - robot.eef_id: end-effector link index
    
    Interface:
        planner = RobotOMPLPlanner(robot, obstacles)
        planner.set_planner("AITstar")
        res, path = planner.plan(start_config, goal_config, planning_time)
        planner.execute(path)
    """
    
    def __init__(self, robot, obstacles=None, collision_margin=0.02, ignore_base=True, robot_urdf=None):
        """
        Initialize OMPL planner with visible collision checking.
        
        Args:
            robot: Robot object (FrankaRobot, Lite6Robot, etc.)
            obstacles: List of obstacle body IDs
            collision_margin: Safety distance from obstacles (meters)
            ignore_base: Ignore base link collisions
            robot_urdf: (Ignored, for compatibility with old interface)
        """
        self.robot = robot
        self.obstacles = obstacles or []
        self.collision_margin = collision_margin
        self.ignore_base = ignore_base
        
        # Auto-detect DOF from robot
        self.joint_ids = robot.arm_controllable_joints
        self.n_joints = len(self.joint_ids)
        # ── NEW: identify gripper joints (all controllable joints beyond the arm) ──
        all_ctrl = robot.controllable_joints          # full list parsed in FrankaRobot
        self.gripper_joint_ids = all_ctrl[self.n_joints:]   # joints 8,9 = fingers
        # Snapshot gripper pose RIGHT NOW (before any planning moves the robot)
        # This captures "fingers closed around box" or "fingers open" — whatever
        # state they're in when planning is requested.
        self._snapshot_gripper_pose()

        # Save initial robot state to restore after planning
        self.saved_state = None
        
        print(f"\n{'='*60}")
        print(f"OMPL PLANNER INITIALIZATION (VISIBLE COLLISION CHECKING)")
        print(f"{'='*60}")
        print(f"Robot ID: {robot.id}")
        print(f"DOF: {self.n_joints}")
        print(f"Joints: {self.joint_ids}")
        print(f"Obstacles: {len(self.obstacles)}")
        print(f"Collision margin: {collision_margin}m")
        print(f"Note: Robot will be moved during planning for collision checks")
        
        # ============================================
        # OMPL STATE SPACE SETUP
        # ============================================
        self.space = ob.RealVectorStateSpace(self.n_joints)
        
        # Set joint limits
        bounds = ob.RealVectorBounds(self.n_joints)
        for i in range(self.n_joints):
            bounds.setLow(i, robot.arm_lower_limits[i])
            bounds.setHigh(i, robot.arm_upper_limits[i])
        self.space.setBounds(bounds)
        
        # Create SimpleSetup
        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        
        # Configure space information
        si = self.ss.getSpaceInformation()
        # si.setStateValidityCheckingResolution(0.01)
        si.setStateValidityCheckingResolution(0.005) 

        
        # Planner will be set via set_planner()
        self.planner = None
        self.planner_type = None
        
        print(f"✓ OMPL planner initialized (call set_planner() to select algorithm)")
        print(f"{'='*60}\n")
    
    def is_state_valid(self, state):
        """
        Check if configuration is collision-free by moving the visible robot.
        Gripper is frozen at its pre-planning pose throughout.
        """
        joint_angles = [state[i] for i in range(self.n_joints)]
        
        # Set arm joints
        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(self.robot.id, joint_id, joint_angles[i])
        
        # ── NEW: freeze gripper so it's always checked at the correct geometry ──
        self._apply_frozen_gripper()
        
        p.performCollisionDetection()
        
        if self._check_self_collision():
            return False
        
        for obs_id in self.obstacles:
            closest_points = p.getClosestPoints(
                self.robot.id, obs_id,
                distance=self.collision_margin
            )
            if closest_points:
                for pt in closest_points:
                    robot_link = pt[3]
                    distance = pt[8]
                    if self.ignore_base and robot_link == -1:
                        continue
                    if distance < self.collision_margin:
                        return False
        
        return True
    # def _save_robot_state(self):
    #     """Save arm + gripper joint state"""
    #     self.saved_state = [p.getJointState(self.robot.id, j)[0] for j in self.joint_ids]
    #     # ── NEW: also save gripper so restore doesn't drop finger pose ──
    #     self.saved_gripper_state = [
    #         p.getJointState(self.robot.id, j)[0] for j in self.gripper_joint_ids
    #     ]

    # def _restore_robot_state(self):
    #     """Restore arm + gripper joint state"""
    #     if self.saved_state is not None:
    #         for i, joint_id in enumerate(self.joint_ids):
    #             p.resetJointState(self.robot.id, joint_id, self.saved_state[i])
    #     # ── NEW: restore gripper too ──
    #     if hasattr(self, 'saved_gripper_state'):
    #         for jid, angle in zip(self.gripper_joint_ids, self.saved_gripper_state):
    #             p.resetJointState(self.robot.id, jid, angle)
    
    def _check_self_collision(self):
        """Check robot self-collision"""
        contacts = p.getContactPoints(self.robot.id, self.robot.id)
        
        for contact in contacts:
            link1 = contact[3]
            link2 = contact[4]
            
            # Ignore base link collisions if flag is set
            if self.ignore_base and (link1 == -1 or link2 == -1):
                continue
            
            # Filter out adjacent link contacts (these are normal)
            if abs(link1 - link2) > 1:  # Non-adjacent links colliding
                return True
        
        return False
    
    def _debug_collision_state(self, config):
        """Debug helper to print collision information for a configuration"""
        # Set robot to test configuration
        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(self.robot.id, joint_id, config[i])
        
        p.performCollisionDetection()
        
        # Check self-collision
        contacts = p.getContactPoints(self.robot.id, self.robot.id)
        if contacts:
            for contact in contacts:
                link1 = contact[3]
                link2 = contact[4]
                if self.ignore_base and (link1 == -1 or link2 == -1):
                    continue
                if abs(link1 - link2) > 1:
                    print(f"    ✗ Self-collision: link {link1} <-> link {link2}")
                    return
        
        # Check collision with obstacles
        for obs_id in self.obstacles:
            closest_points = p.getClosestPoints(
                self.robot.id, obs_id, 
                distance=self.collision_margin
            )
            
            if closest_points:
                for pt in closest_points:
                    robot_link = pt[3]
                    distance = pt[8]
                    
                    if self.ignore_base and robot_link == -1:
                        continue
                    
                    if distance < self.collision_margin:
                        link_name = p.getJointInfo(self.robot.id, robot_link)[12].decode('utf-8') if robot_link >= 0 else "base"
                        print(f"    ✗ Collision: robot link '{link_name}' (id={robot_link}) <-> obstacle {obs_id}, distance={distance:.3f}m (margin={self.collision_margin}m)")
                        return
    
    def set_planner(self, planner_name):
        """
        Set OMPL planner algorithm.
        
        Args:
            planner_name: "AITstar", "RRTstar", "RRTConnect", "RRT", "PRM", etc.
        """
        si = self.ss.getSpaceInformation()
        
        if planner_name == "AITstar":
            self.planner = og.AITstar(si)
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(si)
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(si)
        elif planner_name == "RRT":
            self.planner = og.RRT(si)
        elif planner_name == "PRM":
            self.planner = og.PRM(si)
        elif planner_name == "InformedRRTstar":
            self.planner = og.InformedRRTstar(si)
        elif planner_name == "BITstar":
            self.planner = og.BITstar(si)
        elif planner_name == "ABITstar":
            self.planner = og.ABITstar(si)
        else:
            print(f"Warning: Planner '{planner_name}' not recognized, using AITstar")
            self.planner = og.AITstar(si)
            planner_name = "AITstar"
        
        self.ss.setPlanner(self.planner)
        self.planner_type = planner_name
        print(f"✓ Planner set to: {planner_name}")
    
    def plan(self, start_config, goal_config, planning_time=10.0):
        """
        Plan a path from start to goal configuration(s).

        Supports two calling conventions:

        1. **Single goal** (original / backward-compatible)::

               success, path = planner.plan(start, goal_config)

        2. **Multiple goals** (new) – pass a list of joint configurations.
           They are tried in the given order (best-first when produced by
           :func:`sample_ik_candidates`).  Planning stops as soon as one
           goal succeeds::

               candidates = sample_ik_candidates(robot, planner, pos, orn,
                                                  q_start=start)
               success, path = planner.plan(start, candidates)

        For each goal the method first attempts a fast **RRTConnect** solve
        (``planning_time / 2``).  If RRTConnect fails it retries with the
        planner set via :meth:`set_planner` using the full ``planning_time``.
        This two-stage fallback mirrors a MoveIt-style pipeline and
        significantly reduces "path not found" failures.

        Args:
            start_config:  List of joint angles (start).
            goal_config:   Either a flat list of joint angles (single goal) or
                           a list of such lists (multiple goals).
            planning_time: Max planning time **per goal attempt** in seconds.

        Returns:
            ``(success, path)`` – ``(True, list_of_configs)`` on success,
            ``(False, None)`` on failure.
        """
        if self.planner is None:
            print("✗ Error: No planner set. Call set_planner() first.")
            return False, None

        # ── Normalise goal_config to a list of configs ────────────────────────
        # A single config is a flat list/array of numbers; every element is a
        # scalar (int/float/numpy scalar).  Multiple configs is a list-of-lists
        # or list-of-arrays; the first element is itself a sequence.
        if (len(goal_config) > 0 and
                isinstance(goal_config[0], (list, np.ndarray))):
            # List of configs → multiple goals
            goal_configs = list(goal_config)
        else:
            # Flat list of numbers → single goal
            goal_configs = [goal_config]

        # Save current robot state to restore after planning
        self._save_robot_state()

        # ── Validate start ────────────────────────────────────────────────────
        if not self.is_state_valid_list(start_config):
            print("✗ Start configuration is in collision!")
            self._restore_robot_state()
            return False, None

        si = self.ss.getSpaceInformation()
        # Minimum time for the RRTConnect feasibility probe (seconds).
        _MIN_PROBE_TIME = 5.0

        for goal_idx, gc in enumerate(goal_configs):
            print(f"\nPlanning attempt {goal_idx + 1}/{len(goal_configs)}")
            print(f"  Start: {[f'{x:.2f}' for x in start_config]}")
            print(f"  Goal:  {[f'{x:.2f}' for x in gc]}")

            if not self.is_state_valid_list(gc):
                print(f"  ✗ Goal {goal_idx + 1} is in collision, skipping")
                continue

            # Build OMPL start / goal states
            ompl_start = ob.State(self.space)
            for i in range(self.n_joints):
                ompl_start[i] = start_config[i]

            ompl_goal = ob.State(self.space)
            for i in range(self.n_joints):
                ompl_goal[i] = gc[i]

            # ── Stage 1: fast RRTConnect feasibility probe ─────────────────
            # Skip the probe if the primary planner is already RRTConnect.
            path_list = None
            if self.planner_type != "RRTConnect":
                probe_planner = og.RRTConnect(si)
                self.ss.clear()
                self.ss.setPlanner(probe_planner)
                self.ss.setStartAndGoalStates(ompl_start, ompl_goal)
                probe_time = max(planning_time / 2.0, _MIN_PROBE_TIME)
                print(f"  [Stage 1] RRTConnect probe ({probe_time:.1f}s)...")
                t0      = time.time()
                solved  = self.ss.solve(probe_time)
                elapsed = time.time() - t0
                if solved:
                    self.ss.simplifySolution()
                    path_obj = self.ss.getSolutionPath()
                    path_obj.interpolate()
                    path_list = [[path_obj.getState(i)[j]
                                  for j in range(self.n_joints)]
                                 for i in range(path_obj.getStateCount())]
                    print(f"  ✓ RRTConnect found path: {len(path_list)} wp "
                          f"in {elapsed:.2f}s")
                else:
                    print(f"  ✗ RRTConnect probe failed ({elapsed:.2f}s)")
                # Restore primary planner for Stage 2
                self.ss.setPlanner(self.planner)

            # ── Stage 2: primary planner (if Stage 1 failed) ──────────────
            if path_list is None:
                self.ss.clear()
                self.ss.setPlanner(self.planner)
                self.ss.setStartAndGoalStates(ompl_start, ompl_goal)
                print(f"  [Stage 2] {self.planner_type} ({planning_time:.1f}s)...")
                t0      = time.time()
                solved  = self.ss.solve(planning_time)
                elapsed = time.time() - t0
                if solved:
                    self.ss.simplifySolution()
                    path_obj = self.ss.getSolutionPath()
                    path_obj.interpolate()
                    path_list = [[path_obj.getState(i)[j]
                                  for j in range(self.n_joints)]
                                 for i in range(path_obj.getStateCount())]
                    print(f"  ✓ {self.planner_type} found path: "
                          f"{len(path_list)} wp in {elapsed:.2f}s")
                else:
                    print(f"  ✗ {self.planner_type} failed ({elapsed:.2f}s)")

            if path_list is not None:
                self._restore_robot_state()
                print(f"✓ Path found via goal {goal_idx + 1}/{len(goal_configs)}: "
                      f"{len(path_list)} waypoints")
                return True, path_list

        # All goals exhausted
        self._restore_robot_state()
        print(f"✗ No path found for any of {len(goal_configs)} goal(s)")
        return False, None

    def plan_to_pose(self, start_config, pos, base_orn,
                     planning_time=10.0,
                     num_seeds_per_yaw=15,
                     pos_tol=0.01,
                     orn_tol=0.05,
                     top_n=5):
        """
        High-level API: plan to an end-effector pose using cube X/Y yaw only.

        This method combines IK sampling (:func:`sample_ik_candidates`) with
        multi-goal planning (:meth:`plan`) in a single call.  It generates
        exactly **two yaw variants** (grasp aligned with cube X axis and cube
        Y axis) and tries the best-scored IK candidates until planning
        succeeds.

        Args:
            start_config:      Current joint configuration (list of floats,
                               length = DOF).
            pos:               Target EEF position [x, y, z].
            base_orn:          Base orientation quaternion [x, y, z, w].  The
                               two yaw variants are this and a 90° rotation
                               about world Z.
            planning_time:     Max time per planning attempt (seconds).
            num_seeds_per_yaw: IK seeds tried per yaw variant (default 15).
            pos_tol:           FK position tolerance in metres (default 1 cm).
            orn_tol:           FK orientation tolerance in radians (default
                               ~2.9°).
            top_n:             Max IK candidates passed to planner (default 5).

        Returns:
            ``(success, path)`` – same as :meth:`plan`.

        Example::

            # Grasp a cube; try both cube-X and cube-Y yaw automatically
            success, path = planner.plan_to_pose(
                start_config=robot.get_arm_joints(),
                pos=cube_pos + [0, 0, 0.15],   # pre-grasp offset
                base_orn=p.getQuaternionFromEuler([np.pi, 0, cube_yaw]),
            )
        """
        print(f"\n{'='*60}")
        print(f"plan_to_pose  pos={[f'{x:.3f}' for x in pos]}")
        print(f"  Generating IK candidates (X & Y yaw, {num_seeds_per_yaw} "
              f"seeds/yaw, top_n={top_n})...")
        print(f"{'='*60}")

        candidates = sample_ik_candidates(
            self.robot, self, pos, base_orn,
            q_start=start_config,
            num_seeds_per_yaw=num_seeds_per_yaw,
            pos_tol=pos_tol,
            orn_tol=orn_tol,
            top_n=top_n,
        )

        if not candidates:
            print("✗ plan_to_pose: IK sampling produced no candidates")
            return False, None

        return self.plan(start_config, candidates, planning_time=planning_time)
    
    def execute(self, path, dt=1/240, steps_per_waypoint=100):
        """
        Execute planned path on the robot with position control.
        
        Args:
            path: List of joint configurations
            dt: Simulation timestep
            steps_per_waypoint: Steps to reach each waypoint
        """
        if path is None or len(path) == 0:
            print("✗ No path to execute")
            return
        
        print(f"\nExecuting path with {len(path)} waypoints...")
        
        for i, waypoint in enumerate(path):
            # Set joint targets
            for j, joint_id in enumerate(self.joint_ids):
                p.setJointMotorControl2(
                    self.robot.id,
                    joint_id,
                    p.POSITION_CONTROL,
                    waypoint[j],
                    maxVelocity=getattr(self.robot, 'max_velocity', 3.0),
                    force=200
                )
             # Get current end-effector position
            eef_state = p.getLinkState(self.robot.id, self.robot.eef_id)
            current_eef_pos = eef_state[0]  # World position of the link

            # Visualize trajectory
            visualise_eef_traj(current_eef_pos)
                
            # Simulate motion to waypoint
            for _ in range(steps_per_waypoint):
                p.stepSimulation()
                time.sleep(dt)
                
                # Early exit if close enough
                current = [p.getJointState(self.robot.id, j)[0] for j in self.joint_ids]
                if np.max(np.abs(np.array(current) - np.array(waypoint))) < 0.01:
                    break
            
            if i % max(1, len(path) // 10) == 0:
                print(f"  Progress: {i+1}/{len(path)} waypoints")
        
        print("✓ Path execution completed")
    
    def is_state_valid_list(self, config):
        """Helper to check if a config list is valid"""
        state = ob.State(self.space)
        for i in range(self.n_joints):
            state[i] = config[i]
        return self.is_state_valid(state)
    
    def _save_robot_state(self):
        """Save current robot joint state"""
        self.saved_state = [p.getJointState(self.robot.id, j)[0] for j in self.joint_ids]
    
    def _restore_robot_state(self):
        """Restore saved robot joint state"""
        if self.saved_state is not None:
            for i, joint_id in enumerate(self.joint_ids):
                p.resetJointState(self.robot.id, joint_id, self.saved_state[i])
    
    # def set_collision_filter(self, robot, obstacle_id, link_a=-1, link_b=-1):
    #     """Dummy method for compatibility with old interface (does nothing in visible collision mode)"""
    #     pass
    # ── NEW HELPERS ────────────────────────────────────────────────────────────

    def _snapshot_gripper_pose(self):
        """
        Capture current gripper finger positions once before planning starts.
        These are frozen throughout planning — fingers won't open or close.
        Called in __init__ and can be called again before plan() if gripper
        state changes between planning calls.
        """
        self.frozen_gripper_angles = [
            p.getJointState(self.robot.id, jid)[0]
            for jid in self.gripper_joint_ids
        ]
        print(f"✓ Gripper frozen at: {[f'{a:.4f}' for a in self.frozen_gripper_angles]}"
              f"  (joints {self.gripper_joint_ids})")

    def _apply_frozen_gripper(self):
        """
        Reset gripper joints to their frozen angles.
        Called inside is_state_valid() so every collision sample
        has the gripper in the correct pose.
        """
        for jid, angle in zip(self.gripper_joint_ids, self.frozen_gripper_angles):
            p.resetJointState(self.robot.id, jid, angle)
    def _apply_frozen_gripper_printing(self):
        """
        Reset gripper joints to their frozen angles.
        Called inside is_state_valid() so every collision sample
        has the gripper in the correct pose.
        """
        for jid, angle in zip(self.gripper_joint_ids, self.frozen_gripper_angles):
            p.resetJointState(self.robot.id, jid, angle)
        print(f"Planning Done , restoring the freezed gripper pose: {[f'{a:.4f}' for a in self.frozen_gripper_angles]}")

    def cleanup(self):
        """No-op — original planner has no ghost bodies to remove."""
        pass

class FrankaRobot:
    """
    Simple Franka Panda robot class for testing.
    Follows same interface pattern as Lite6Robot for compatibility with RobotOMPLPlanner.
    """
    
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 11  # Franka EEF link index
        self.arm_num_dofs = 7
        self.arm_rest_poses = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.max_velocity = 3.0
        self.id = None
    
    def load(self):
        """Load Franka Panda URDF"""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.id = p.loadURDF(
            "franka_panda/panda.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
        )
        
        self._parse_joint_info()
        self._print_debug_info()
    
    def _parse_joint_info(self):
        """Parse joint information from URDF"""
        jointInfo = namedtuple(
            "jointInfo",
            ["id", "name", "type", "lowerLimit", "upperLimit", 
             "maxForce", "maxVelocity", "controllable"],
        )
        
        self.joints = []
        self.controllable_joints = []
        
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            
            if controllable:
                self.controllable_joints.append(jointID)
            
            self.joints.append(
                jointInfo(
                    jointID, jointName, jointType,
                    jointLowerLimit, jointUpperLimit,
                    jointMaxForce, jointMaxVelocity,
                    controllable,
                )
            )
        
        # First 7 joints are arm, rest are gripper
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [
            ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)
        ]
    
    def _print_debug_info(self):
        """Print debug information"""
        print("\n" + "="*60)
        print("FRANKA ROBOT DEBUG INFO")
        print("="*60)
        jtype_names = {0: 'REVOLUTE', 1: 'PRISMATIC', 4: 'FIXED'}
        for j in self.joints:
            jtype = jtype_names.get(j.type, str(j.type))
            ctrl = "CTRL" if j.controllable else "    "
            print(f"  Joint {j.id:2d}: {j.name:<35s} ({jtype:<9s}) [{ctrl}]")
        
        print(f"\nArm controllable joints: {self.arm_controllable_joints}")
        print(f"Arm DOF: {self.arm_num_dofs}")
        print(f"EEF link index: {self.eef_id}")
        
        eef_state = p.getLinkState(self.id, self.eef_id)
        print(f"EEF position: {eef_state[0]}")
        print(f"EEF orientation (quat): {eef_state[1]}")
        print("="*60 + "\n")
    
    def get_current_ee_position(self):
        """Get current end-effector pose"""
        eef_state = p.getLinkState(self.id, self.eef_id)
        return eef_state[0], eef_state[1]
