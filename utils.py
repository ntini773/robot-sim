import pybullet as p

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
    
    # for i, joint_id in enumerate(robot.arm_controllable_joints):
    #         p.resetJointState(robot.id, joint_id, waypoint[0])