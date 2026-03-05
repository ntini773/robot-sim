"""
Test OMPL planner with Lite6 robot (6-DOF)

Demonstrates drop-in integration with existing Lite6Robot class.
Same RobotOMPLPlanner works for both 6-DOF and 7-DOF robots.
Uses solve_ik_collision_free for robust IK with collision checking.
"""

import sys
import pybullet as p
import pybullet_data
import numpy as np
import time

# Import Lite6Robot from existing pick_and_place script
from pick_and_place_xarm6_gripper import Lite6Robot

# Import planner
from planner import RobotOMPLPlanner, solve_ik_collision_free


def create_box(position, size=[0.1,0.1,0.1], color=[1,0,0,1]):
    """Create a box obstacle"""
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )
    return body


def draw_sphere(position, color=[1,0,0], radius=0.03):
    """Draw a sphere marker"""
    visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color + [1]
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=position
    )


def setup_environment():
    """Setup table, plane, and obstacles"""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useMaximalCoordinates=True)
    table_id = p.loadURDF(
        "table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0])
    )
    p.setTimeStep(1/240)
    
    # Floating obstacle
    box1 = create_box(
        position=[0.5, 0, 0.9],
        size=[0.1, 0.08, 0.1],
        color=[1, 0, 0, 1]
    )
    
    obstacles = [plane_id, table_id, box1]
    return table_id, obstacles


def visualize_path(robot, path):
    """Draw the end-effector path"""
    prev_ee = None
    
    for waypoint in path:
        # Set robot to waypoint
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            p.resetJointState(robot.id, joint_id, waypoint[i])
        
        # Get EE position
        ee_pos = p.getLinkState(robot.id, robot.eef_id)[0]
        
        # Draw line from previous to current
        if prev_ee is not None:
            p.addUserDebugLine(prev_ee, ee_pos, [0, 1, 0], 3)
        
        prev_ee = ee_pos


def test_planner(planner_type, robot, planner, obstacles,
                 start_pos, start_orn, goal_pos, goal_orn):
    """Test a specific planner"""
    print(f"\n{'='*60}")
    print(f"TESTING {planner_type} PLANNER")
    print(f"{'='*60}\n")
    
    # Set planner algorithm
    planner.set_planner(planner_type)
    
    # Solve IK with collision checking
    print("Solving IK for start...")
    start_config = solve_ik_collision_free(robot, planner, start_pos, start_orn)
    
    if start_config is None:
        print(f"✗ {planner_type}: Failed to find collision-free start IK")
        return None, None, None
    
    print("Solving IK for goal...")
    goal_config = solve_ik_collision_free(robot, planner, goal_pos, goal_orn)
    
    if goal_config is None:
        print(f"✗ {planner_type}: Failed to find collision-free goal IK")
        return None, None, None
    
    print(f"\n✓ IK solutions found")
    print(f"  Start: {[f'{x:.2f}' for x in start_config]}")
    print(f"  Goal:  {[f'{x:.2f}' for x in goal_config]}")
    
    # Move robot to start
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        p.resetJointState(robot.id, joint_id, start_config[i])
    
    # Draw start/goal EE positions
    start_ee = p.getLinkState(robot.id, robot.eef_id)[0]
    goal_ee_temp = list(goal_config)  # Temp set to visualize
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        p.resetJointState(robot.id, joint_id, goal_ee_temp[i])
    goal_ee = p.getLinkState(robot.id, robot.eef_id)[0]
    
    # Restore to start
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        p.resetJointState(robot.id, joint_id, start_config[i])
    
    draw_sphere(start_ee, [0, 1, 0])  # Green = start
    draw_sphere(goal_ee, [1, 0, 0])   # Red = goal
    
    # Plan
    start_time = time.time()
    success, path = planner.plan(start_config, goal_config, planning_time=10.0)
    planning_time = time.time() - start_time
    
    if success and path:
        # Calculate path length
        path_length = 0
        for i in range(len(path) - 1):
            path_length += np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
        
        print(f"\n{'='*60}")
        print(f"{planner_type} RESULTS")
        print(f"{'='*60}")
        print(f"Planning time: {planning_time:.2f}s")
        print(f"Waypoints: {len(path)}")
        print(f"Path length: {path_length:.3f} rad")
        print(f"{'='*60}\n")
        
        # Visualize planned path
        visualize_path(robot, path)
        
        # Execute path
        print("Executing planned path...")
        planner.execute(path, dt=1/240, steps_per_waypoint=50)
        
        return path, planning_time, path_length
    else:
        print(f"✗ {planner_type} failed to find a path")
        return None, planning_time, None


def main():
    """Test OMPL planning with Lite6 robot"""
    
    # Connect to PyBullet
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.4, cameraYaw=90,
        cameraPitch=-42.6, cameraTargetPosition=[0.0, 0.0, 0.0]
    )
    p.setGravity(0, 0, -9.8)
    
    # Define start and goal poses
    start_pos = [0.4, -0.3, 0.85]
    start_orn = p.getQuaternionFromEuler([3.14, 0, 0])
    goal_pos = [0.4, 0.2, 0.85]
    goal_orn = p.getQuaternionFromEuler([3.14, 0, 0])
    
    # Visualize start and goal positions
    start_marker = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 1, 0, 0.8]  # Green
    )
    start_marker_id = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=start_marker,
        basePosition=start_pos
    )
    
    goal_marker = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.8]  # Red
    )
    goal_marker_id = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=goal_marker,
        basePosition=goal_pos
    )
    
    print(f"\n{'='*60}")
    print(f"TARGET POSES VISUALIZATION")
    print(f"{'='*60}")
    print(f"Start (GREEN):  {start_pos}")
    print(f"Goal (RED):     {goal_pos}")
    print(f"{'='*60}\n")
    
    # Setup environment
    table_id, obstacles = setup_environment()
    
    # Load Lite6 robot (existing class with custom gripper)
    print("\n" + "="*60)
    print("LOADING LITE6 ROBOT")
    print("="*60)
    robot = Lite6Robot([0, 0, 0.62], [0, 0, 0])
    robot.load()
    
    # Disable collision between robot and table
    for j in range(p.getNumJoints(robot.id)):
        p.setCollisionFilterPair(
            robot.id, table_id, j, -1, enableCollision=0
        )
    
    # Create planner (reuse for both algorithms)
    print("\n" + "="*60)
    print("CREATING OMPL PLANNER")
    print("="*60)
    planner = RobotOMPLPlanner(
        robot=robot,
        robot_urdf="./lite-6-updated-urdf/lite_6_new.urdf",
        obstacles=obstacles,
        collision_margin=0.02,
        ignore_base=True
    )
    
    # Disable collision between collision robot and table (robot is mounted on table)
    planner.set_collision_filter(robot, table_id, link_a=-1)
    
    # Test AITstar
    print("\n" + "="*60)
    print("TESTING AITSTAR VS RRTSTAR")
    print("="*60)
    
    aitstar_result = test_planner(
        "AITstar", robot, planner, obstacles,
        start_pos, start_orn, goal_pos, goal_orn
    )
    
    # Wait a moment
    time.sleep(2)
    
    # Test RRTstar (reuse same planner instance)
    rrtstar_result = test_planner(
        "RRTstar", robot, planner, obstacles,
        start_pos, start_orn, goal_pos, goal_orn
    )
    
    # Comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    if aitstar_result[0] and rrtstar_result[0]:
        print(f"\nAITstar:")
        print(f"  Planning time: {aitstar_result[1]:.2f}s")
        print(f"  Waypoints: {len(aitstar_result[0])}")
        print(f"  Path length: {aitstar_result[2]:.3f} rad")
        
        print(f"\nRRTstar:")
        print(f"  Planning time: {rrtstar_result[1]:.2f}s")
        print(f"  Waypoints: {len(rrtstar_result[0])}")
        print(f"  Path length: {rrtstar_result[2]:.3f} rad")
        
        if aitstar_result[2] < rrtstar_result[2]:
            print(f"\n✓ AITstar found shorter path by {rrtstar_result[2] - aitstar_result[2]:.3f} rad")
        else:
            print(f"\n✓ RRTstar found shorter path by {aitstar_result[2] - rrtstar_result[2]:.3f} rad")
    
    print(f"\n{'='*60}\n")
    
    # Keep simulation running
    print("Simulation running. Press Ctrl+C to exit.")
    try:
        while True:
            p.stepSimulation()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nShutting down...")
        p.disconnect()


if __name__ == "__main__":
    main()
