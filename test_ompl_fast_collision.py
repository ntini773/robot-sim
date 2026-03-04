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
        obstacles=[plane],
        robot_urdf="franka_panda/panda.urdf",
        robot_base_pos=robot_base_pos
    )
    
    print("\n" + "="*60)
    print("FAST COLLISION MODE:")
    print("- Visible robot: FROZEN during planning")
    print("- Collision robot: INVISIBLE (alpha=0)")
    print("- Same physics client: MUCH FASTER")
    print("="*60 + "\n")
    
    # Test 1: Simple joint space movement
    print(f"\n{'='*60}")
    print("TEST 1: Move joint 0 by 0.5 radians")
    print(f"{'='*60}")
    
    start = home_config.copy()
    goal = home_config.copy()
    goal[0] += 0.5  # Move first joint
    
    path = planner.plan(start, goal, planning_time=1.0)
    
    if path:
        print(f"\n✓ Success! Executing path...")
        execute_path(panda, arm_joints, path)
        time.sleep(1)
    else:
        print("\n✗ Planning failed")
    
    # Test 2: Multi-joint movement
    print(f"\n{'='*60}")
    print("TEST 2: Move multiple joints")
    print(f"{'='*60}")
    
    current = [p.getJointState(panda, j)[0] for j in arm_joints]
    goal2 = home_config.copy()
    goal2[1] -= 0.3
    goal2[3] += 0.3
    goal2[5] += 0.5
    
    path2 = planner.plan(current, goal2, planning_time=1.0)
    
    if path2:
        print(f"\n✓ Success! Executing path...")
        execute_path(panda, arm_joints, path2)
    else:
        print("\n✗ Planning failed")
    
    # Test 3: Longer path
    print(f"\n{'='*60}")
    print("TEST 3: Larger movement (stress test)")
    print(f"{'='*60}")
    
    current = [p.getJointState(panda, j)[0] for j in arm_joints]
    goal3 = home_config.copy()
    goal3[0] += 1.0
    goal3[1] -= 0.5
    
    path3 = planner.plan(current, goal3, planning_time=1.0)
    
    if path3:
        print(f"\n✓ Success! Executing path...")
        execute_path(panda, arm_joints, path3)
    else:
        print("\n✗ Planning failed")
    
    # Keep running
    print(f"\n{'='*60}")
    print("Tests complete. Press Ctrl+C to exit")
    print(f"{'='*60}\n")
    
    try:
        while True:
            p.stepSimulation()
            time.sleep(1/240)
    except KeyboardInterrupt:
        p.disconnect()


def execute_path(robot_id, joint_ids, path, steps_per_waypoint=100):
    """Execute a planned path with smooth control"""
    for waypoint in path:
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
