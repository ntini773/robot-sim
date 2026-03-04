"""
OMPL AITstar planner with FAST collision checking
Uses invisible collision-only robot in the same physics client
MUCH faster than separate client approach
"""
import pybullet as p
import pybullet_data
import time
import numpy as np

import ompl.base as ob
import ompl.geometric as og

collision_checks = 0


class FastOMPLPlanner:
    """OMPL planner with fast invisible collision robot"""
    
    def __init__(self, robot_id, joint_ids, joint_limits, obstacles=None, 
                 robot_urdf="franka_panda/panda.urdf", robot_base_pos=[0, 0, 0.5]):
        """
        Args:
            robot_id: PyBullet body ID (main visible robot)
            joint_ids: List of controllable joint indices
            joint_limits: List of (lower, upper) tuples for each joint
            obstacles: List of obstacle body IDs
            robot_urdf: Path to robot URDF
            robot_base_pos: Base position of robot
        """
        self.robot_id = robot_id  # Main visible robot
        self.joint_ids = joint_ids
        self.n_joints = len(joint_ids)
        self.obstacles = obstacles or []
        
        # ============================================
        # CREATE INVISIBLE COLLISION ROBOT
        # ============================================
        print("Creating invisible collision robot...")
        
        # Load collision robot FAR AWAY from camera (won't be in frame)
        # Place at y=1000 to be out of view
        collision_base_pos = [robot_base_pos[0], robot_base_pos[1] + 1000, robot_base_pos[2]]
        
        self.collision_robot_id = p.loadURDF(
            robot_urdf,
            collision_base_pos,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION  # Enable self-collision
        )
        
        # Make collision robot INVISIBLE by setting alpha = 0
        num_joints = p.getNumJoints(self.collision_robot_id)
        
        # Remove visual shapes from base
        p.changeVisualShape(self.collision_robot_id, -1, rgbaColor=[0, 0, 0, 0])
        
        # Remove visual shapes from all links
        for i in range(num_joints):
            p.changeVisualShape(self.collision_robot_id, i, rgbaColor=[0, 0, 0, 0])
        
        print(f"✓ Invisible collision robot created")
        print(f"  - Collision robot ID: {self.collision_robot_id}")
        print(f"  - Position: {collision_base_pos} (far from camera)")
        print(f"  - Visual shapes: HIDDEN (alpha=0)")
        
        # ============================================
        # OMPL SETUP
        # ============================================
        self.space = ob.RealVectorStateSpace(self.n_joints)
        
        # Set joint limits
        bounds = ob.RealVectorBounds(self.n_joints)
        for i, (lower, upper) in enumerate(joint_limits):
            bounds.setLow(i, lower)
            bounds.setHigh(i, upper)
        self.space.setBounds(bounds)
        
        # Setup OMPL
        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        
        # Use AITstar planner
        si = self.ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(0.01)
        planner = og.AITstar(si)
        self.ss.setPlanner(planner)
        
        print(f"✓ OMPL Planner created")
        print(f"  - DOF: {self.n_joints}")
        print(f"  - Obstacles: {len(self.obstacles)}")
    
    def is_state_valid(self, state):
        """Check collision using INVISIBLE collision robot (FAST)"""
        global collision_checks
        collision_checks += 1
        
        # Progress indicator
        if collision_checks % 5000 == 0:
            print(f"  Collision checks: {collision_checks}")
        
        # Extract joint values
        joint_values = [state[i] for i in range(self.n_joints)]
        
        # ============================================
        # Set COLLISION robot (invisible, far away) - NOT visible robot!
        # ============================================
        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(
                self.collision_robot_id,  # Invisible robot
                joint_id, 
                joint_values[i]
            )
        
        # Single collision detection call (fast!)
        p.performCollisionDetection()
        
        # ============================================
        # Check self-collision
        # ============================================
        # self_contacts = p.getContactPoints(
        #     self.collision_robot_id, 
        #     self.collision_robot_id
        # )
        
        # for contact in self_contacts:
        #     link1, link2 = contact[3], contact[4]
        #     # Ignore adjacent links
        #     if abs(link1 - link2) > 1:
        #         return False
        
        # ============================================
        # Check obstacle collision (optional - skip if no obstacles)
        # ============================================
        if self.obstacles:
            for obs_id in self.obstacles:
                contacts = p.getClosestPoints(
                    self.collision_robot_id, 
                    obs_id, 
                    distance=0.02
                )
                
                if contacts and contacts[0][8] < 0.02:  # Within 2cm
                    return False
        
        # Valid! (and visible robot never moved!)
        return True
    
    def plan(self, start_config, goal_config, planning_time=5.0):
        """Plan path from start to goal"""
        global collision_checks
        collision_checks = 0
        
        self.ss.clear()
        
        # Validate start/goal
        print(f"\nStart: {[f'{x:.2f}' for x in start_config]}")
        print(f"Goal:  {[f'{x:.2f}' for x in goal_config]}")
        
        if not self._is_valid_config(start_config):
            print("✗ Start configuration invalid!")
            return None
        print("✓ Start valid")
        
        if not self._is_valid_config(goal_config):
            print("✗ Goal configuration invalid!")
            return None
        print("✓ Goal valid")
        
        # Set start/goal
        start = ob.State(self.space)
        goal = ob.State(self.space)
        
        for i in range(self.n_joints):
            start[i] = start_config[i]
            goal[i] = goal_config[i]
        
        self.ss.setStartAndGoalStates(start, goal)
        
        print(f"Planning (timeout={planning_time}s)...")
        start_time = time.time()
        
        # Solve
        solved = self.ss.solve(planning_time)
        
        elapsed = time.time() - start_time
        
        if solved:
            self.ss.simplifySolution()
            path_obj = self.ss.getSolutionPath()
            
            path = []
            for i in range(path_obj.getStateCount()):
                state = path_obj.getState(i)
                config = [state[j] for j in range(self.n_joints)]
                path.append(config)
            
            print(f"✓ Path found: {len(path)} waypoints")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Collision checks: {collision_checks}")
            print(f"  Rate: {collision_checks/elapsed:.0f} checks/sec")
            return path
        else:
            print("✗ No path found")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Collision checks: {collision_checks}")
            return None
    
    def _is_valid_config(self, config):
        """Check if config is valid"""
        state = ob.State(self.space)
        for i in range(self.n_joints):
            state[i] = config[i]
        return self.is_state_valid(state)


def test_panda_robot():
    """Test with Franka Panda robot"""
    
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(1/240)
    
    # Load environment
    plane = p.loadURDF("plane.urdf")
    
    # Load Panda robot
    robot_base_pos = [0, 0, 0.5]
    panda = p.loadURDF("franka_panda/panda.urdf", robot_base_pos, useFixedBase=True)
    
    # Add obstacle box in front of robot (in workspace)
    # Position it where the robot arm might reach
    obstacle_pos = [0.3, 0, 0.7]  # Front-right of robot, at arm height
    obstacle_size = [0.15, 0.15, 0.1]  # Width, depth, height
    
    collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=obstacle_size
    )
    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=obstacle_size,
        rgbaColor=[1, 0, 0, 0.7]  # Red semi-transparent
    )
    
    obstacle_box = p.createMultiBody(
        baseMass=0,  # Static obstacle
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=obstacle_pos
    )
    
    print(f"\n{'='*60}")
    print(f"Obstacle created at {obstacle_pos}")
    print(f"Size: {[2*s for s in obstacle_size]} (full dimensions)")
    print(f"{'='*60}")
    
    # Get arm joint info (first 7 joints are the arm)
    num_joints = p.getNumJoints(panda)
    print(f"\n{'='*60}")
    print(f"Panda Robot - Total joints: {num_joints}")
    print(f"{'='*60}")
    
    arm_joints = []
    arm_limits = []
    
    for i in range(7):  # Panda has 7 arm joints
        info = p.getJointInfo(panda, i)
        joint_name = info[1].decode('utf-8')
        joint_type = info[2]
        lower = info[8]
        upper = info[9]
        
        if joint_type == p.JOINT_REVOLUTE:
            arm_joints.append(i)
            arm_limits.append((lower, upper))
            print(f"Joint {i}: {joint_name:20s} [{lower:+.2f}, {upper:+.2f}]")
    
    print(f"{'='*60}\n")
    
    # Set to home position
    home_config = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    
    for i, joint_id in enumerate(arm_joints):
        p.resetJointState(panda, joint_id, home_config[i])
    
    # Let physics settle
    for _ in range(100):
        p.stepSimulation()
    
    # Create FAST planner with invisible collision robot
    planner = FastOMPLPlanner(
        robot_id=panda,
        joint_ids=arm_joints,
        joint_limits=arm_limits,
        obstacles=[plane, obstacle_box],  # Include obstacle in collision checking
        robot_urdf="franka_panda/panda.urdf",
        robot_base_pos=robot_base_pos
    )
    
    print("\n" + "="*60)
    print("FAST COLLISION MODE:")
    print("- Visible robot: FROZEN during planning")
    print("- Collision robot: INVISIBLE (alpha=0)")
    print("- Same physics client: MUCH FASTER")
    print("="*60 + "\n")
    
    # ============================================
    # Define start and goal in END-EFFECTOR SPACE (Cartesian coordinates)
    # ============================================
    print(f"\n{'='*60}")
    print("DEFINING START AND GOAL IN END-EFFECTOR SPACE")
    print(f"{'='*60}")
    
    # Start position: LEFT side of obstacle
    start_ee_pos = [0.3, -0.3, 0.8]  # x, y, z in meters
    start_ee_orn = p.getQuaternionFromEuler([3.14, 0, 0])  # Pointing down
    
    # Goal position: RIGHT side of obstacle  
    goal_ee_pos = [0.3, 0.3, 0.8]  # x, y, z in meters
    goal_ee_orn = p.getQuaternionFromEuler([3.14, 0, 0])  # Pointing down
    
    print(f"Start EE position: {[f'{x:.2f}' for x in start_ee_pos]}")
    print(f"Goal EE position:  {[f'{x:.2f}' for x in goal_ee_pos]}")
    print(f"Obstacle at:       {[f'{x:.2f}' for x in obstacle_pos]}")
    
    # ============================================
    # Use Inverse Kinematics to get joint configurations
    # ============================================
    print(f"\nSolving inverse kinematics...")
    
    # IK for start position
    start_config = list(p.calculateInverseKinematics(
        panda,
        11,  # End-effector link index for Panda
        start_ee_pos,
        start_ee_orn,
        maxNumIterations=100,
        residualThreshold=1e-5
    ))[:7]  # Take only first 7 joints (arm joints)
    
    # IK for goal position
    goal_config = list(p.calculateInverseKinematics(
        panda,
        11,  # End-effector link index for Panda
        goal_ee_pos,
        goal_ee_orn,
        maxNumIterations=100,
        residualThreshold=1e-5
    ))[:7]  # Take only first 7 joints (arm joints)
    
    print(f"✓ IK solved")
    print(f"  Start joints: {[f'{x:.2f}' for x in start_config]}")
    print(f"  Goal joints:  {[f'{x:.2f}' for x in goal_config]}")
    
    # ============================================
    # Verify IK solutions by setting robot and checking positions
    # ============================================
    # Set robot to start configuration
    for i, joint_id in enumerate(arm_joints):
        p.resetJointState(panda, joint_id, start_config[i])
    
    for _ in range(50):
        p.stepSimulation()
    
    # Verify start position
    start_ee_state = p.getLinkState(panda, 11)
    start_ee_actual = start_ee_state[0]
    start_error = np.linalg.norm(np.array(start_ee_pos) - np.array(start_ee_actual))
    
    # Set to goal configuration
    for i, joint_id in enumerate(arm_joints):
        p.resetJointState(panda, joint_id, goal_config[i])
    
    for _ in range(50):
        p.stepSimulation()
    
    # Verify goal position
    goal_ee_state = p.getLinkState(panda, 11)
    goal_ee_actual = goal_ee_state[0]
    goal_error = np.linalg.norm(np.array(goal_ee_pos) - np.array(goal_ee_actual))
    
    print(f"\nIK Verification:")
    print(f"  Start error: {start_error*1000:.2f}mm")
    print(f"  Goal error:  {goal_error*1000:.2f}mm")
    
    # Reset to start position
    for i, joint_id in enumerate(arm_joints):
        p.resetJointState(panda, joint_id, start_config[i])
    
    # ============================================
    # Create visual markers for start and goal positions
    # ============================================
    start_marker = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.05,
        rgbaColor=[0, 1, 0, 0.8]  # Green for start
    )
    goal_marker = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.05,
        rgbaColor=[0, 0, 1, 0.8]  # Blue for goal
    )
    
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=start_marker,
        basePosition=start_ee_pos
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=goal_marker,
        basePosition=goal_ee_pos
    )
    
    print(f"\n{'='*60}")
    print("VISUALIZATION:")
    print(f"  Green sphere: START EE at {[f'{x:.2f}' for x in start_ee_pos]}")
    print(f"  Blue sphere:  GOAL EE at {[f'{x:.2f}' for x in goal_ee_pos]}")
    print(f"  Red box:      OBSTACLE at {obstacle_pos}")
    print(f"{'='*60}")
    
    # Wait for user to see the setup
    print("\nSetup complete. Press Enter to start planning...")
    input()
    
    # Main test: Navigate from left to right around obstacle
    print(f"\n{'='*60}")
    print("MAIN TEST: Navigate LEFT → RIGHT around obstacle")
    print(f"{'='*60}")
    
    path = planner.plan(start_config, goal_config, planning_time=10.0)
    
    if path:
        print(f"\n✓ Success! Path found. Executing...")
        execute_path(panda, arm_joints, path, steps_per_waypoint=150)
    else:
        print("\n✗ Planning failed - no collision-free path found")
    
    # Keep running for observation
    print(f"\n{'='*60}")
    print("Path execution complete!")
    print("End-effector moved from GREEN to BLUE around RED obstacle")
    print("Press Ctrl+C to exit")
    print(f"{'='*60}\n")
    
    try:
        while True:
            p.stepSimulation()
            time.sleep(1/240)
    except KeyboardInterrupt:
        p.disconnect()


def execute_path(robot_id, joint_ids, path, steps_per_waypoint=100):
    """Execute a planned path with smooth control"""
    print(f"Executing {len(path)} waypoints...")
    for idx, waypoint in enumerate(path):
        if idx % 5 == 0:
            print(f"  Waypoint {idx+1}/{len(path)}")
        
        for i, joint_id in enumerate(joint_ids):
            p.setJointMotorControl2(
                robot_id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=waypoint[i],
                force=300,
                maxVelocity=0.5,
                positionGain=0.03,
                velocityGain=1.0
            )
        
        for _ in range(steps_per_waypoint):
            p.stepSimulation()
            time.sleep(1/240)


if __name__ == "__main__":
    test_panda_robot()
