"""
Minimal OMPL AITstar planner test with Panda robot
Tests basic planning functionality without complex scene setup
"""
import pybullet as p
import pybullet_data
import time
import numpy as np

import ompl.base as ob
import ompl.geometric as og

resetting = 0


class SimpleOMPLPlanner:
    """Minimal OMPL planner for testing with HIDDEN collision checking"""
    
    def __init__(self, robot_id, joint_ids, joint_limits, obstacles=None, robot_urdf="franka_panda/panda.urdf", robot_base_pos=[0, 0, 0.5]):
        """
        Args:
            robot_id: PyBullet body ID (main visible robot)
            joint_ids: List of controllable joint indices
            joint_limits: List of (lower, upper) tuples for each joint
            obstacles: List of obstacle body IDs in main client
            robot_urdf: Path to robot URDF
            robot_base_pos: Base position of robot
        """
        self.robot_id = robot_id  # Main visible robot
        self.joint_ids = joint_ids
        self.n_joints = len(joint_ids)
        self.obstacles = obstacles or []
        
        # ============================================
        # CREATE HIDDEN COLLISION CLIENT
        # ============================================
        print("Creating hidden collision environment...")
        self.collision_client = p.connect(p.DIRECT)  # No GUI, hidden
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.collision_client)
        
        # Load shadow robot in hidden client
        self.shadow_robot_id = p.loadURDF(
            robot_urdf,
            robot_base_pos,
            useFixedBase=True,
            physicsClientId=self.collision_client
        )
        
        # Load shadow obstacles in hidden client
        self.shadow_obstacles = []
        for obs_id in obstacles:
            # Get obstacle info from main client
            base_pos, base_orn = p.getBasePositionAndOrientation(obs_id)
            
            # Determine obstacle type and load in shadow client
            # For this example, we assume it's the plane
            shadow_obs = p.loadURDF(
                "plane.urdf",
                base_pos,
                base_orn,
                physicsClientId=self.collision_client
            )
            self.shadow_obstacles.append(shadow_obs)
        
        print(f"✓ Shadow collision environment created (HIDDEN)")
        print(f"  - Shadow robot ID: {self.shadow_robot_id}")
        print(f"  - Shadow obstacles: {len(self.shadow_obstacles)}")
        
        # ============================================
        # OMPL SETUP
        # ============================================
        # Create state space
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
        """Check if configuration is collision-free using HIDDEN shadow robot"""
        global resetting
        resetting += 1
        if resetting % 1000 == 0:
            print(f"Collision checks: {resetting}")
        
        # Extract joint values
        joint_values = [state[i] for i in range(self.n_joints)]
        
        # ============================================
        # Set SHADOW robot (in hidden client) - NOT visible robot!
        # ============================================
        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(
                self.shadow_robot_id, 
                joint_id, 
                joint_values[i],
                physicsClientId=self.collision_client  # HIDDEN client
            )
        
        p.performCollisionDetection(physicsClientId=self.collision_client)
        
        # ============================================
        # Check self-collision in HIDDEN client
        # ============================================
        self_contacts = p.getContactPoints(
            self.shadow_robot_id, 
            self.shadow_robot_id,
            physicsClientId=self.collision_client
        )
        
        for contact in self_contacts:
            link1, link2 = contact[3], contact[4]
            # Ignore adjacent links
            if abs(link1 - link2) > 1:
                return False
        
        # ============================================
        # Check obstacle collision in HIDDEN client
        # ============================================
        for shadow_obs_id in self.shadow_obstacles:
            contacts = p.getClosestPoints(
                self.shadow_robot_id, 
                shadow_obs_id, 
                distance=0.02,
                physicsClientId=self.collision_client
            )
            
            if contacts and contacts[0][8] < 0.02:  # Within 2cm
                return False
        
        # Valid! (and visible robot never moved!)
        return True
    
    def plan(self, start_config, goal_config, planning_time=5.0):
        """Plan path from start to goal"""
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
        
        # Solve
        solved = self.ss.solve(planning_time)
        
        if solved:
            self.ss.simplifySolution()
            path_obj = self.ss.getSolutionPath()
            
            path = []
            for i in range(path_obj.getStateCount()):
                state = path_obj.getState(i)
                config = [state[j] for j in range(self.n_joints)]
                path.append(config)
            
            print(f"✓ Path found: {len(path)} waypoints")
            print(f"Total collision checks: {resetting}")
            return path
        else:
            print("✗ No path found")
            print(f"Total collision checks: {resetting}")
            return None
    
    def _is_valid_config(self, config):
        """Check if config is valid"""
        state = ob.State(self.space)
        for i in range(self.n_joints):
            state[i] = config[i]
        return self.is_state_valid(state)
    
    def __del__(self):
        """Cleanup: disconnect shadow collision client"""
        try:
            p.disconnect(physicsClientId=self.collision_client)
            print("✓ Shadow collision environment cleaned up")
        except:
            pass


def test_panda_robot():
    """Test with Franka Panda robot"""
    
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(1/240)
    
    # Load environment
    plane = p.loadURDF("plane.urdf")
    # table = p.loadURDF("table/table.urdf", [0.5, 0, 0])
    
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
    
    # Create planner with HIDDEN collision environment
    planner = SimpleOMPLPlanner(
        robot_id=panda,
        joint_ids=arm_joints,
        joint_limits=arm_limits,
        obstacles=[plane],
        robot_urdf="franka_panda/panda.urdf",
        robot_base_pos=robot_base_pos
    )
    
    print("\n" + "="*60)
    print("NOTE: Watch the GUI - the visible robot should NOT move during planning!")
    print("Only collision checks happen in hidden environment")
    print("="*60 + "\n")
    
    # Test 1: Simple joint space movement
    print(f"\n{'='*60}")
    print("TEST 1: Move joint 0 by 0.5 radians")
    print(f"{'='*60}")
    
    start = home_config.copy()
    goal = home_config.copy()
    goal[0] += 0.5  # Move first joint
    
    path = planner.plan(start, goal, planning_time=10.0)
    
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
    
    path2 = planner.plan(current, goal2, planning_time=10.0)
    
    if path2:
        print(f"\n✓ Success! Executing path...")
        execute_path(panda, arm_joints, path2)
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