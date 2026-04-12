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

def interactive_camera_helper():
    """
    Run in GUI mode. Fly around with mouse, press Enter to print current camera as _MP_CAMERAS entry.
    Middle mouse = pan, scroll = zoom, right drag = orbit
    """
    print("Fly to desired view using mouse, then press Enter in terminal to capture.")
    print("Ctrl+C to stop.\n")
    
    while True:
        input(">>> Press Enter to capture current GUI camera position...")
        
        # Read PyBullet's current GUI camera state
        cam_info = p.getDebugVisualizerCamera()
        
        # cam_info fields:
        # [0]  width
        # [1]  height
        # [2]  viewMatrix        (16 floats, column-major)
        # [3]  projectionMatrix  (16 floats)
        # [4]  cameraUp          (x,y,z)
        # [5]  cameraForward     (x,y,z)
        # [6]  horizontal        (x,y,z)
        # [7]  yaw               (degrees)  ← GUI orbit yaw
        # [8]  pitch             (degrees)  ← GUI orbit pitch
        # [9]  dist              (float)    ← distance from target
        # [10] target            (x,y,z)   ← orbit center / look-at point
        
        target   = cam_info[11]   # (x, y, z)
        dist     = cam_info[10]
        yaw      = cam_info[8]
        pitch    = cam_info[9]
        cam_up   = cam_info[4]
        forward  = cam_info[5]
        
        # Reconstruct eye position from target + forward + dist
        forward_arr = np.array(forward)
        target_arr  = np.array(target)
        eye = target_arr - forward_arr * dist
        
        print("\n--- Copy this into _MP_CAMERAS ---")
        print(f"""{{
    "name": "thirdperson_camXX",
    "eye":    [{eye[0]:.4f}, {eye[1]:.4f}, {eye[2]:.4f}],
    "target": [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}],
    "up":     [{cam_up[0]:.4f}, {cam_up[1]:.4f}, {cam_up[2]:.4f}],
}},""")
        print(f"(yaw={yaw:.1f}°, pitch={pitch:.1f}°, dist={dist:.3f})\n")