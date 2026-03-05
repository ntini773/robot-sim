"""
Robot-Agnostic OMPL Motion Planner

Interface similar to pybullet_ompl for clean separation of planning and execution.

Usage:
    from planner import RobotOMPLPlanner, FrankaRobot, solve_ik_collision_free
    
    # Setup robot and environment
    robot = FrankaRobot([0, 0, 0.63], [0, 0, 0])
    robot.load()
    
    # Create planner
    planner = RobotOMPLPlanner(robot, robot_urdf, obstacles)
    planner.set_planner("AITstar")  # or "RRTstar", "RRTConnect"
    
    # Solve IK with collision checking
    start = solve_ik_collision_free(robot, planner, start_pos, start_orn)
    goal = solve_ik_collision_free(robot, planner, goal_pos, goal_orn)
    
    # Plan path
    res, path = planner.plan(start, goal, planning_time=10.0)
    
    # Execute path
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


def solve_ik_collision_free(robot, planner, pos, orn, max_attempts=50):
    """
    Solve IK with collision checking and bounds validation, trying multiple random seeds.
    
    Args:
        robot: Robot object with arm_controllable_joints, etc.
        planner: RobotOMPLPlanner instance for collision checking
        pos: Target [x, y, z] position
        orn: Target [x, y, z, w] quaternion
        max_attempts: Max IK attempts with different seeds
        
    Returns:
        Joint configuration (list) or None if no collision-free solution found
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
        
        # CRITICAL: Validate bounds (PyBullet IK can violate them)
        q_array = np.array(q)
        if np.any(q_array < lower - 1e-4) or np.any(q_array > upper + 1e-4):
            if attempt == 0:
                print(f"  ⚠ IK solution violates bounds, retrying with different seed...")
            continue
        
        # Clamp to bounds for safety
        q = np.clip(q_array, lower, upper).tolist()
        
        # Check collision
        if planner.is_state_valid_list(q):
            if attempt > 0:
                print(f"  ✓ IK found valid config on attempt {attempt+1}")
            return q
    
    print(f"  ✗ WARNING: No valid IK solution found after {max_attempts} attempts")
    return None


class RobotOMPLPlanner:
    """
    Robot-agnostic OMPL motion planner following pybullet_ompl interface.
    
    Works with any robot class (duck typing) that provides:
    - robot.id: PyBullet body ID
    - robot.arm_controllable_joints: list of joint indices
    - robot.arm_lower_limits: list of lower limits
    - robot.arm_upper_limits: list of upper limits
    - robot.base_pos: [x, y, z] base position
    - robot.eef_id: end-effector link index
    
    Interface:
        planner = RobotOMPLPlanner(robot, robot_urdf, obstacles)
        planner.set_planner("AITstar")
        res, path = planner.plan(start_config, goal_config, planning_time)
        planner.execute(path)
    """
    
    def __init__(self, robot, robot_urdf, obstacles=None, 
                 collision_margin=0.02, ignore_base=True):
        """
        Initialize OMPL planner with invisible collision robot.
        
        Args:
            robot: Robot object (FrankaRobot, Lite6Robot, etc.)
            robot_urdf: Path to robot URDF file
            obstacles: List of obstacle body IDs
            collision_margin: Safety distance from obstacles (meters)
            ignore_base: Ignore base link collisions
        """
        self.robot = robot
        self.robot_urdf = robot_urdf
        self.obstacles = obstacles or []
        self.collision_margin = collision_margin
        self.ignore_base = ignore_base
        
        # Auto-detect DOF from robot
        self.joint_ids = robot.arm_controllable_joints
        self.n_joints = len(self.joint_ids)
        
        print(f"\n{'='*60}")
        print(f"OMPL PLANNER INITIALIZATION")
        print(f"{'='*60}")
        print(f"Robot ID: {robot.id}")
        print(f"DOF: {self.n_joints}")
        print(f"Joints: {self.joint_ids}")
        print(f"Obstacles: {len(self.obstacles)}")
        print(f"Collision margin: {collision_margin}m")
        
        # ============================================
        # CREATE INVISIBLE COLLISION ROBOT (FAST)
        # ============================================
        print(f"\nCreating invisible collision robot and shadow obstacles...")
        
        # Load collision robot FAR AWAY (y + 1000m, out of camera view)
        self.y_offset = 1000.0
        collision_base_pos = [
            robot.base_pos[0], 
            robot.base_pos[1] + self.y_offset, 
            robot.base_pos[2]
        ]
        
        self.collision_robot_id = p.loadURDF(
            robot_urdf,
            collision_base_pos,
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        
        # Make collision robot INVISIBLE (alpha=0 on all links)
        p.changeVisualShape(self.collision_robot_id, -1, rgbaColor=[0, 0, 0, 0])  # Base
        for i in range(p.getNumJoints(self.collision_robot_id)):
            p.changeVisualShape(self.collision_robot_id, i, rgbaColor=[0, 0, 0, 0])
        
        print(f"✓ Invisible collision robot created")
        print(f"  - Collision robot ID: {self.collision_robot_id}")
        print(f"  - Position: y+{self.y_offset}m (out of view)")
        
        # ============================================
        # CREATE SHADOW OBSTACLES (CRITICAL!)
        # ============================================
        # Obstacles must be at same y-position as collision robot for collision detection to work
        self.shadow_obstacles = []
        for obs_id in self.obstacles:
            # Get original obstacle properties
            pos, orn = p.getBasePositionAndOrientation(obs_id)
            
            # Create shadow obstacle at collision robot's y-position
            shadow_pos = [pos[0], pos[1] + self.y_offset, pos[2]]
            
            # Get visual shape info to duplicate obstacle
            visual_data = p.getVisualShapeData(obs_id)
            collision_data = p.getCollisionShapeData(obs_id)
            
            if len(collision_data) > 0:
                col_shape = collision_data[0]
                geom_type = col_shape[2]
                dimensions = col_shape[3]
                
                # Create shadow obstacle based on geometry type
                if geom_type == p.GEOM_BOX:
                    col_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=dimensions)
                    vis_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=dimensions, 
                                                       rgbaColor=[0, 0, 0, 0])  # Invisible
                elif geom_type == p.GEOM_PLANE:
                    # For plane, just use loadURDF at new position
                    shadow_obs_id = p.loadURDF("plane.urdf", shadow_pos, orn, useMaximalCoordinates=True)
                    self.shadow_obstacles.append(shadow_obs_id)
                    continue
                elif geom_type == p.GEOM_MESH:
                    # For complex meshes (like table), reload URDF
                    # This is a heuristic - works for most standard URDFs
                    try:
                        # Try to reload the URDF (works for table, etc.)
                        shadow_obs_id = p.loadURDF(p.getBodyInfo(obs_id)[1].decode(), 
                                                    shadow_pos, orn)
                        # Make shadow invisible
                        p.changeVisualShape(shadow_obs_id, -1, rgbaColor=[0, 0, 0, 0])
                        for j in range(p.getNumJoints(shadow_obs_id)):
                            p.changeVisualShape(shadow_obs_id, j, rgbaColor=[0, 0, 0, 0])
                        self.shadow_obstacles.append(shadow_obs_id)
                        continue
                    except:
                        # Fallback: create bounding box
                        aabb_min, aabb_max = p.getAABB(obs_id)
                        box_size = [(aabb_max[i] - aabb_min[i]) / 2.0 for i in range(3)]
                        col_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_size)
                        vis_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=box_size, 
                                                           rgbaColor=[0, 0, 0, 0])
                else:
                    # Default: use bounding box
                    aabb_min, aabb_max = p.getAABB(obs_id)
                    box_size = [(aabb_max[i] - aabb_min[i]) / 2.0 for i in range(3)]
                    col_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_size)
                    vis_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=box_size, 
                                                       rgbaColor=[0, 0, 0, 0])
                
                # Create the shadow body
                shadow_obs_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col_shape_id,
                    baseVisualShapeIndex=vis_shape_id,
                    basePosition=shadow_pos,
                    baseOrientation=orn
                )
                self.shadow_obstacles.append(shadow_obs_id)
        
        print(f"✓ Created {len(self.shadow_obstacles)} shadow obstacles at y+{self.y_offset}m")
        print(f"  - Shadow obstacle IDs: {self.shadow_obstacles}")
    
    def set_collision_filter(self, robot,body_a, link_a=-1):
        """
        Disable collision between the invisible collision robot and a specific body.
        Useful for disabling robot-table collision if robot is mounted on table.
        
        Args:
            body_a: Body ID to filter collision with
            link_a: Link ID of body_a (-1 for base)
        """
        for j in range(p.getNumJoints(self.collision_robot_id) + 1):
            link_idx = j - 1  # -1 for base, then 0, 1, 2...
            p.setCollisionFilterPair(
                self.collision_robot_id, body_a, 
                link_idx, link_a, 
                enableCollision=0
            )
        print(f"✓ Disabled collision between collision robot and body {body_a}")
        
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
        si.setStateValidityCheckingResolution(0.01)
        
        # Planner will be set via set_planner()
        self.planner = None
        self.planner_type = None
        
        print(f"✓ OMPL planner initialized (call set_planner() to select algorithm)")
        print(f"{'='*60}\n")
    
    def is_state_valid(self, state):
        """
        Check if configuration is collision-free using INVISIBLE collision robot.
        Main robot never moves during planning!
        """
        # Extract joint angles from OMPL state
        joint_angles = [state[i] for i in range(self.n_joints)]
        
        # Set COLLISION robot to test configuration (invisible, far away)
        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(self.collision_robot_id, joint_id, joint_angles[i])
        
        p.performCollisionDetection()
        
        # Check self-collision
        if self._check_self_collision():
            return False
        
        # Check collision with obstacles
        for obs_id in self.obstacles:
            closest_points = p.getClosestPoints(
                self.collision_robot_id, obs_id, 
                distance=self.collision_margin
            )
            
            if closest_points:
                for pt in closest_points:
                    robot_link = pt[3]
                    distance = pt[8]
                    
                    # Skip base link if we're ignoring it
                    if self.ignore_base and robot_link == -1:
                        continue
                    
                    if distance < self.collision_margin:
                        return False
        
        return True
    
    def _check_self_collision(self):
        """Check robot self-collision using collision robot"""
        contacts = p.getContactPoints(self.collision_robot_id, self.collision_robot_id)
        
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
        else:
            print(f"Warning: Planner '{planner_name}' not recognized, using AITstar")
            self.planner = og.AITstar(si)
            planner_name = "AITstar"
        
        self.ss.setPlanner(self.planner)
        self.planner_type = planner_name
        print(f"✓ Planner set to: {planner_name}")
    
    def plan(self, start_config, goal_config, planning_time=10.0):
        """
        Plan a path from start to goal configuration.
        
        Args:
            start_config: List of joint angles (start)
            goal_config: List of joint angles (goal)
            planning_time: Max planning time in seconds
            
        Returns:
            (success, path): Tuple of (bool, list of configs or None)
                            - (True, path) if planning succeeds
                            - (False, None) if planning fails
        """
        if self.planner is None:
            print("✗ Error: No planner set. Call set_planner() first.")
            return False, None
        
        # Clear previous planning data
        self.ss.clear()
        
        # Validate start/goal
        if not self.is_state_valid_list(start_config):
            print("✗ Start configuration is in collision!")
            return False, None
        
        if not self.is_state_valid_list(goal_config):
            print("✗ Goal configuration is in collision!")
            return False, None
        
        # Set start state
        start = ob.State(self.space)
        for i in range(self.n_joints):
            start[i] = start_config[i]
        
        # Set goal state
        goal = ob.State(self.space)
        for i in range(self.n_joints):
            goal[i] = goal_config[i]
        
        self.ss.setStartAndGoalStates(start, goal)
        
        print(f"\nPlanning with {self.planner_type}...")
        print(f"  Start: {[f'{x:.2f}' for x in start_config]}")
        print(f"  Goal:  {[f'{x:.2f}' for x in goal_config]}")
        
        # Solve
        start_time = time.time()
        solved = self.ss.solve(planning_time)
        elapsed = time.time() - start_time
        
        if solved:
            # Simplify path
            self.ss.simplifySolution()
            
            # Extract path
            path_obj = self.ss.getSolutionPath()
            
            # Convert to list of configurations
            path_list = []
            for i in range(path_obj.getStateCount()):
                state = path_obj.getState(i)
                config = [state[j] for j in range(self.n_joints)]
                path_list.append(config)
            
            print(f"✓ Path found: {len(path_list)} waypoints in {elapsed:.2f}s")
            return True, path_list
        else:
            print(f"✗ No path found within {planning_time}s (elapsed: {elapsed:.2f}s)")
            return False, None
    
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
    
    def __del__(self):
        """Cleanup: remove invisible collision robot"""
        try:
            if hasattr(self, 'collision_robot_id'):
                p.removeBody(self.collision_robot_id)
                print("Cleaned up collision robot")
        except:
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
