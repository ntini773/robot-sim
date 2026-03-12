"""
Ghost Robot OMPL Motion Planner - Uses a hidden duplicate robot for collision checking.

Identical interface to RobotOMPLPlanner (planner.py) — drop-in replacement.
The key difference: a "ghost" robot is loaded at a far-off position and used
exclusively for collision checking. The main robot's state is NEVER touched
during planning or state validation.

Usage (identical to planner.py):
    from planner_with_collision_robot import RobotOMPLPlanner

    planner = RobotOMPLPlanner(
        robot=robot,
        robot_urdf="./lite-6-updated-urdf/lite_6_new.urdf",
        obstacles=[plane_id, table_id, cube_id],
        collision_margin=0.02,
        ignore_base=True
    )
    planner.set_planner("AITStar")
    success, path = planner.plan(start_config, goal_config, planning_time=10.0)
"""

import pybullet as p
import numpy as np
import time

import ompl.base as ob
import ompl.geometric as og

# ── re-export helpers that callers import from planner.py ─────────────────────
from planner import visualise_eef_traj, solve_ik, solve_ik_collision_free  # noqa: F401

# Offset applied to every ghost obstacle / ghost robot so they live far from
# the real scene. Large enough that no real object is nearby.
_GHOST_OFFSET = np.array([-5.0, 0.0, 0.0])


class RobotOMPLPlanner:
    """
    Drop-in replacement for planner.RobotOMPLPlanner.

    Internally loads a second "ghost" copy of the robot (and clones every
    obstacle as a static ghost) at a fixed world offset. All collision queries
    are performed on the ghost world — the main robot joint state is never
    modified during planning.

    Public interface is identical to the original class.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, robot, obstacles=None, collision_margin=0.02,
                 ignore_base=True, robot_urdf=None, ghost_visible=True):
        """
        Args:
            robot        : Main robot object (Lite6Robot / FrankaRobot …)
            obstacles    : List of *main-world* PyBullet body IDs
            collision_margin : Safety clearance in metres
            ignore_base  : Ignore base-link collisions (index -1)
            robot_urdf   : Path to URDF used to load the ghost robot.
                           If None, tries robot.urdf_path then raises.
        """
        self.robot = robot
        self.obstacles = obstacles or []
        self.collision_margin = collision_margin
        self.ignore_base = ignore_base
        self.ghost_visible = ghost_visible

        # Resolve URDF path ------------------------------------------------
        if robot_urdf is None:
            robot_urdf = getattr(robot, 'urdf_path', None)
        if robot_urdf is None:
            raise ValueError(
                "robot_urdf must be supplied (robot has no .urdf_path attribute)."
            )
        self._robot_urdf = robot_urdf

        # Arm joint bookkeeping (mirrors main robot) -----------------------
        self.joint_ids = robot.arm_controllable_joints   # indices on MAIN robot
        self.n_joints  = len(self.joint_ids)

        # Gripper joints on the MAIN robot (for snapshot only)
        all_ctrl = robot.controllable_joints
        self.gripper_joint_ids = all_ctrl[self.n_joints:]

        # ------------------------------------------------------------------
        # Build ghost world
        # ------------------------------------------------------------------
        self._ghost_robot_id    = None   # PyBullet ID of ghost robot
        self._ghost_obstacle_ids = []    # PyBullet IDs of ghost obstacles
        self._ghost_joint_ids   = []     # arm joint indices on ghost robot
        self._ghost_gripper_ids = []     # gripper joint indices on ghost robot

        self._load_ghost_robot()
        self._load_ghost_obstacles()

        # Snapshot gripper BEFORE any planning so collision checks use the
        # correct finger geometry.
        self._snapshot_gripper_pose()

        # Saved arm state (unused in ghost variant but kept for API compat)
        self.saved_state = None

        # ------------------------------------------------------------------
        # OMPL state space
        # ------------------------------------------------------------------
        self.space = ob.RealVectorStateSpace(self.n_joints)
        bounds = ob.RealVectorBounds(self.n_joints)
        for i in range(self.n_joints):
            bounds.setLow(i,  robot.arm_lower_limits[i])
            bounds.setHigh(i, robot.arm_upper_limits[i])
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.is_state_valid)
        )
        si = self.ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(0.005)

        self.planner      = None
        self.planner_type = None

        print(f"\n{'='*60}")
        print(f"GHOST-ROBOT OMPL PLANNER")
        print(f"{'='*60}")
        print(f"Main robot ID  : {robot.id}")
        print(f"Ghost robot ID : {self._ghost_robot_id}")
        print(f"Ghost offset   : {_GHOST_OFFSET.tolist()}")
        print(f"DOF            : {self.n_joints}")
        print(f"Obstacles      : {len(self.obstacles)}  →  "
              f"ghost copies: {self._ghost_obstacle_ids}")
        print(f"Collision margin: {collision_margin} m")
        print(f"Main robot state is NEVER modified during planning.")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Ghost world setup
    # ------------------------------------------------------------------

    def _load_ghost_robot(self):
        """Load an invisible copy of the robot at _GHOST_OFFSET."""
        base_pos = np.array(self.robot.base_pos) + _GHOST_OFFSET
        base_orn = self.robot.base_ori  # quaternion

        ghost_id = p.loadURDF(
            self._robot_urdf,
            base_pos.tolist(),
            base_orn,
            useFixedBase=True,
        )

        # Make every link visible or invisible based on flag
        alpha = 1.0 if self.ghost_visible else 0.0
        for link in range(-1, p.getNumJoints(ghost_id)):
            p.changeVisualShape(ghost_id, link, rgbaColor=[0.3, 0.3, 1.0, alpha])

        # Identify arm + gripper joint indices on the ghost robot.
        # We assume joint ordering is identical to the main robot.
        controllable = []
        for i in range(p.getNumJoints(ghost_id)):
            info = p.getJointInfo(ghost_id, i)
            if info[2] != p.JOINT_FIXED:
                controllable.append(info[0])

        self._ghost_joint_ids   = controllable[:self.n_joints]
        self._ghost_gripper_ids = controllable[self.n_joints:]
        self._ghost_robot_id    = ghost_id

        # Mirror main robot's current arm pose onto ghost
        self._sync_ghost_arm_to_main()

        print(f"  Ghost robot loaded: id={ghost_id}  "
              f"arm joints={self._ghost_joint_ids}  "
              f"gripper joints={self._ghost_gripper_ids}")

    def _load_ghost_obstacles(self):
        """
        For every obstacle body in self.obstacles, create a static ghost copy
        at the same relative geometry but shifted by _GHOST_OFFSET.

        Strategy: read each link's AABB and collision shape, then create a
        matching MultiBody at the offset position.  Because obstacles are
        typically simple primitives (plane, table, cube, cylinder) this works
        reliably.  We use p.getCollisionShapeData to reconstruct shape.
        """
        self._ghost_obstacle_ids = []

        for obs_id in self.obstacles:
            ghost_obs_ids = self._clone_body_as_ghost(obs_id)
            self._ghost_obstacle_ids.extend(ghost_obs_ids)

    def _clone_body_as_ghost(self, body_id):
        """
        Clone a PyBullet body as a static ghost shifted by _GHOST_OFFSET.

        Returns a list of ghost body IDs (one per link that has a collision
        shape, since PyBullet can only create multi-shape bodies via URDF).
        For simple single-link bodies the list has one element.
        """
        ghost_ids = []
        num_joints = p.getNumJoints(body_id)
        base_pos_world, base_orn_world = p.getBasePositionAndOrientation(body_id)

        # PyBullet getCollisionShapeData does NOT accept -1 for base link.
        # Index 0 queries the base shape; actual base world pose used for it.
        # For multi-link bodies also iterate links 1..n-1.
        links_to_clone = list(range(max(num_joints, 1)))  # at minimum [0]

        for link_idx in links_to_clone:
            try:
                shapes = p.getCollisionShapeData(body_id, link_idx)
            except Exception:
                continue
            if not shapes:
                continue

            # World pose of this link
            if link_idx == 0:
                pos_world, orn_world = base_pos_world, base_orn_world
            else:
                link_state = p.getLinkState(body_id, link_idx)
                pos_world  = link_state[4]
                orn_world  = link_state[5]

            ghost_pos = (np.array(pos_world) + _GHOST_OFFSET).tolist()

            for shape in shapes:
                shape_type      = shape[2]
                shape_dims      = shape[3]   # depends on type
                local_pos       = shape[5]
                local_orn       = shape[6]

                # Build collision shape
                col_id = self._create_collision_shape(shape_type, shape_dims)
                if col_id < 0:
                    continue

                # Visual shape — same geometry, semi-transparent red for debug
                vis_id = self._create_visual_shape(shape_type, shape_dims)

                ghost_body = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=ghost_pos,
                    baseOrientation=orn_world,
                )
                ghost_ids.append(ghost_body)

        if not ghost_ids:
            # Fallback: use AABB-based box
            aabb_min, aabb_max = p.getAABB(body_id)
            dims = [(aabb_max[i] - aabb_min[i]) / 2.0 for i in range(3)]
            center = [(aabb_max[i] + aabb_min[i]) / 2.0 for i in range(3)]
            ghost_pos = (np.array(center) + _GHOST_OFFSET).tolist()

            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=dims)
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=dims,
                                         rgbaColor=[1, 0, 0, 0.3])
            ghost_body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=ghost_pos,
            )
            ghost_ids.append(ghost_body)

        return ghost_ids

    @staticmethod
    def _create_collision_shape(shape_type, dims):
        if shape_type == p.GEOM_BOX:
            return p.createCollisionShape(p.GEOM_BOX, halfExtents=dims)
        elif shape_type == p.GEOM_SPHERE:
            return p.createCollisionShape(p.GEOM_SPHERE, radius=dims[0])
        elif shape_type == p.GEOM_CYLINDER:
            return p.createCollisionShape(p.GEOM_CYLINDER,
                                          radius=dims[0], height=dims[1])
        elif shape_type == p.GEOM_CAPSULE:
            return p.createCollisionShape(p.GEOM_CAPSULE,
                                          radius=dims[0], height=dims[1])
        elif shape_type == p.GEOM_PLANE:
            return p.createCollisionShape(p.GEOM_PLANE,
                                          planeNormal=[0, 0, 1])
        elif shape_type == p.GEOM_MESH:
            # Mesh shapes require a filename; fall back to a thin box
            return p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=[0.5, 0.5, 0.005])
        return -1

    @staticmethod
    def _create_visual_shape(shape_type, dims):
        alpha = 1.0 if self.ghost_visible else 0.15
        color = [0.5, 0.5, 1.0, alpha]   # blue ghost obstacles
        if shape_type == p.GEOM_BOX:
            return p.createVisualShape(p.GEOM_BOX,
                                       halfExtents=dims, rgbaColor=color)
        elif shape_type == p.GEOM_SPHERE:
            return p.createVisualShape(p.GEOM_SPHERE,
                                       radius=dims[0], rgbaColor=color)
        elif shape_type == p.GEOM_CYLINDER:
            return p.createVisualShape(p.GEOM_CYLINDER,
                                       radius=dims[0], length=dims[1],
                                       rgbaColor=color)
        elif shape_type == p.GEOM_CAPSULE:
            return p.createVisualShape(p.GEOM_CAPSULE,
                                       radius=dims[0], length=dims[1],
                                       rgbaColor=color)
        elif shape_type == p.GEOM_PLANE:
            return p.createVisualShape(p.GEOM_BOX,
                                       halfExtents=[5, 5, 0.001],
                                       rgbaColor=color)
        return p.createVisualShape(p.GEOM_BOX,
                                   halfExtents=[0.05, 0.05, 0.05],
                                   rgbaColor=color)

    # ------------------------------------------------------------------
    # Obstacle management (called by main script between phases)
    # ------------------------------------------------------------------

    def add_obstacle(self, obs_id):
        """Add a new main-world obstacle and create its ghost copy."""
        if obs_id not in self.obstacles:
            self.obstacles.append(obs_id)
            ghost_ids = self._clone_body_as_ghost(obs_id)
            self._ghost_obstacle_ids.extend(ghost_ids)
            # Keep a mapping so we can remove by main ID
            if not hasattr(self, '_obs_to_ghost'):
                self._obs_to_ghost = {}
            self._obs_to_ghost[obs_id] = ghost_ids
            print(f"  + Obstacle {obs_id} added  →  ghost ids {ghost_ids}")

    def remove_obstacle(self, obs_id):
        """Remove an obstacle from collision checking and delete its ghost."""
        if obs_id in self.obstacles:
            self.obstacles.remove(obs_id)

        if not hasattr(self, '_obs_to_ghost'):
            return

        ghost_ids = self._obs_to_ghost.pop(obs_id, [])
        for gid in ghost_ids:
            if gid in self._ghost_obstacle_ids:
                self._ghost_obstacle_ids.remove(gid)
            try:
                p.removeBody(gid)
            except Exception:
                pass
        print(f"  - Obstacle {obs_id} removed  (ghost ids {ghost_ids} deleted)")

    # ------------------------------------------------------------------
    # Gripper snapshot / freeze  (API identical to original planner)
    # ------------------------------------------------------------------

    def _snapshot_gripper_pose(self):
        """
        Capture current gripper angles from the MAIN robot.
        Applied to the ghost robot during every collision check.
        """
        self.frozen_gripper_angles = [
            p.getJointState(self.robot.id, jid)[0]
            for jid in self.gripper_joint_ids
        ]
        # Mirror onto ghost immediately
        for jid, angle in zip(self._ghost_gripper_ids,
                               self.frozen_gripper_angles):
            p.resetJointState(self._ghost_robot_id, jid, angle)

        print(f"✓ Gripper frozen at: "
              f"{[f'{a:.4f}' for a in self.frozen_gripper_angles]}"
              f"  (main joints {self.gripper_joint_ids})")

    def _apply_frozen_gripper(self):
        """Apply frozen gripper angles to the GHOST robot only."""
        for jid, angle in zip(self._ghost_gripper_ids,
                               self.frozen_gripper_angles):
            p.resetJointState(self._ghost_robot_id, jid, angle)

    def _apply_frozen_gripper_printing(self):
        """Same as _apply_frozen_gripper but with a log line."""
        self._apply_frozen_gripper()
        print(f"Planning done — restoring frozen gripper pose: "
              f"{[f'{a:.4f}' for a in self.frozen_gripper_angles]}")

    # ------------------------------------------------------------------
    # State validity  (operates entirely on ghost robot)
    # ------------------------------------------------------------------

    def is_state_valid(self, state):
        """Collision check using the GHOST robot. Main robot untouched."""
        joint_angles = [state[i] for i in range(self.n_joints)]

        # Move ghost arm
        for i, ghost_jid in enumerate(self._ghost_joint_ids):
            p.resetJointState(self._ghost_robot_id, ghost_jid, joint_angles[i])

        # Freeze ghost gripper
        self._apply_frozen_gripper()

        p.performCollisionDetection()

        # Self-collision on ghost
        if self._check_self_collision():
            return False

        # Ghost robot vs ghost obstacles
        for ghost_obs_id in self._ghost_obstacle_ids:
            closest = p.getClosestPoints(
                self._ghost_robot_id, ghost_obs_id,
                distance=self.collision_margin
            )
            if closest:
                for pt in closest:
                    robot_link = pt[3]
                    distance   = pt[8]
                    if self.ignore_base and robot_link == -1:
                        continue
                    if distance < self.collision_margin:
                        return False

        return True

    def is_state_valid_list(self, config):
        """Helper: check a plain list config."""
        state = ob.State(self.space)
        for i in range(self.n_joints):
            state[i] = config[i]
        return self.is_state_valid(state)

    def _check_self_collision(self):
        """Self-collision check on ghost robot."""
        contacts = p.getContactPoints(self._ghost_robot_id,
                                      self._ghost_robot_id)
        for contact in contacts:
            link1 = contact[3]
            link2 = contact[4]
            if self.ignore_base and (link1 == -1 or link2 == -1):
                continue
            if abs(link1 - link2) > 1:
                return True
        return False

    def _debug_collision_state(self, config):
        """Debug helper — sets ghost to config and prints collision info."""
        for i, ghost_jid in enumerate(self._ghost_joint_ids):
            p.resetJointState(self._ghost_robot_id, ghost_jid, config[i])
        self._apply_frozen_gripper()
        p.performCollisionDetection()

        contacts = p.getContactPoints(self._ghost_robot_id,
                                      self._ghost_robot_id)
        for contact in contacts:
            link1, link2 = contact[3], contact[4]
            if self.ignore_base and (link1 == -1 or link2 == -1):
                continue
            if abs(link1 - link2) > 1:
                print(f"    ✗ Ghost self-collision: link {link1} <-> link {link2}")
                return

        for ghost_obs_id in self._ghost_obstacle_ids:
            closest = p.getClosestPoints(
                self._ghost_robot_id, ghost_obs_id,
                distance=self.collision_margin
            )
            if closest:
                for pt in closest:
                    robot_link = pt[3]
                    distance   = pt[8]
                    if self.ignore_base and robot_link == -1:
                        continue
                    if distance < self.collision_margin:
                        link_name = (
                            p.getJointInfo(self._ghost_robot_id,
                                           robot_link)[12].decode('utf-8')
                            if robot_link >= 0 else "base"
                        )
                        print(f"    ✗ Ghost collision: link '{link_name}' "
                              f"(id={robot_link}) <-> ghost obs {ghost_obs_id}, "
                              f"dist={distance:.3f}m  (margin={self.collision_margin}m)")
                        return

    # ------------------------------------------------------------------
    # Planner selection
    # ------------------------------------------------------------------

    def set_planner(self, planner_name):
        si = self.ss.getSpaceInformation()

        planners = {
            "AITstar":         og.AITstar,
            "RRTstar":         og.RRTstar,
            "RRTConnect":      og.RRTConnect,
            "RRT":             og.RRT,
            "PRM":             og.PRM,
            "InformedRRTstar": og.InformedRRTstar,
            "BITstar":         og.BITstar,
            "ABITstar":        og.ABITstar,
        }
        cls = planners.get(planner_name)
        if cls is None:
            print(f"Warning: '{planner_name}' not recognised, using AITstar")
            cls = og.AITstar
            planner_name = "AITstar"

        self.planner = cls(si)
        self.ss.setPlanner(self.planner)
        self.planner_type = planner_name
        print(f"✓ Planner set to: {planner_name}")

    # ------------------------------------------------------------------
    # Planning  (main robot state NEVER touched)
    # ------------------------------------------------------------------

    def plan(self, start_config, goal_config, planning_time=10.0):
        if self.planner is None:
            print("✗ No planner set. Call set_planner() first.")
            return False, None

        # _save/_restore kept for API compatibility but do nothing here
        self._save_robot_state()

        self.ss.clear()

        if not self.is_state_valid_list(start_config):
            print("✗ Start configuration is in collision!")
            self._restore_robot_state()
            return False, None

        if not self.is_state_valid_list(goal_config):
            print("✗ Goal configuration is in collision!")
            self._restore_robot_state()
            return False, None

        start = ob.State(self.space)
        goal  = ob.State(self.space)
        for i in range(self.n_joints):
            start[i] = start_config[i]
            goal[i]  = goal_config[i]

        self.ss.setStartAndGoalStates(start, goal)

        print(f"\nPlanning with {self.planner_type} (ghost collision world)...")
        print(f"  Start: {[f'{x:.2f}' for x in start_config]}")
        print(f"  Goal : {[f'{x:.2f}' for x in goal_config]}")

        t0     = time.time()
        solved = self.ss.solve(planning_time)
        elapsed = time.time() - t0

        self._restore_robot_state()   # no-op in ghost variant

        if solved:
            self.ss.simplifySolution()
            path_obj = self.ss.getSolutionPath()
            path_obj.interpolate()

            path_list = []
            for i in range(path_obj.getStateCount()):
                state = path_obj.getState(i)
                path_list.append([state[j] for j in range(self.n_joints)])

            print(f"✓ Path found: {len(path_list)} waypoints in {elapsed:.2f}s")
            return True, path_list
        else:
            print(f"✗ No path found within {planning_time}s")
            return False, None

    # ------------------------------------------------------------------
    # Execution  (identical to original — drives MAIN robot)
    # ------------------------------------------------------------------

    def execute(self, path, dt=1/240, steps_per_waypoint=100):
        if not path:
            print("✗ No path to execute")
            return

        print(f"\nExecuting path with {len(path)} waypoints...")
        for i, waypoint in enumerate(path):
            for j, joint_id in enumerate(self.joint_ids):   # MAIN robot joints
                p.setJointMotorControl2(
                    self.robot.id,
                    joint_id,
                    p.POSITION_CONTROL,
                    waypoint[j],
                    maxVelocity=getattr(self.robot, 'max_velocity', 3.0),
                    force=200,
                )

            eef_state = p.getLinkState(self.robot.id, self.robot.eef_id)
            visualise_eef_traj(eef_state[0])

            for _ in range(steps_per_waypoint):
                p.stepSimulation()
                time.sleep(dt)
                current = [p.getJointState(self.robot.id, j)[0]
                           for j in self.joint_ids]
                if np.max(np.abs(np.array(current) - np.array(waypoint))) < 0.01:
                    break

            if i % max(1, len(path) // 10) == 0:
                print(f"  Progress: {i+1}/{len(path)} waypoints")

        print("✓ Path execution completed")

    # ------------------------------------------------------------------
    # Save / restore  (no-op in ghost variant — kept for API compat)
    # ------------------------------------------------------------------

    def _save_robot_state(self):
        """No-op: main robot state is never modified."""
        pass

    def _restore_robot_state(self):
        """No-op: main robot state is never modified."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sync_ghost_arm_to_main(self):
        """Mirror main robot's current arm pose onto the ghost."""
        for main_jid, ghost_jid in zip(self.joint_ids, self._ghost_joint_ids):
            angle = p.getJointState(self.robot.id, main_jid)[0]
            p.resetJointState(self._ghost_robot_id, ghost_jid, angle)

    def cleanup(self):
        """Remove ghost bodies from simulation (call when done with planner)."""
        try:
            p.removeBody(self._ghost_robot_id)
        except Exception:
            pass
        for gid in self._ghost_obstacle_ids:
            try:
                p.removeBody(gid)
            except Exception:
                pass
        print("Ghost world cleaned up.")
