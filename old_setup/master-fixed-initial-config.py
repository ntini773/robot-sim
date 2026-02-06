import os
import pybullet as p
import pybullet_data
import math
import time
import random
from collections import namedtuple
import cv2
import numpy as np
import json


class UR5Robotiq85:
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0, 0.085]
        self.max_velocity = 3

    def load(self):
        self.id = p.loadURDF(
            "./urdf/ur5_robotiq_85.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
        )
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

    def __parse_joint_info__(self):
        jointInfo = namedtuple(
            "jointInfo",
            [
                "id",
                "name",
                "type",
                "lowerLimit",
                "upperLimit",
                "maxForce",
                "maxVelocity",
                "controllable",
            ],
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
                    jointID,
                    jointName,
                    jointType,
                    jointLowerLimit,
                    jointUpperLimit,
                    jointMaxForce,
                    jointMaxVelocity,
                    controllable,
                )
            )
        self.arm_controllable_joints = self.controllable_joints[: self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][
            : self.arm_num_dofs
        ]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][
            : self.arm_num_dofs
        ]
        self.arm_joint_ranges = [
            ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)
        ]

    def __setup_mimic_joints__(self):
        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }
        self.mimic_parent_id = [
            joint.id for joint in self.joints if joint.name == mimic_parent_name
        ][0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints
            if joint.name in mimic_children_names
        }
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(
                self.id,
                self.mimic_parent_id,
                self.id,
                joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def move_arm_ik(self, target_pos, target_orn):
        joint_poses = p.calculateInverseKinematics(
            self.id,
            self.eef_id,
            target_pos,
            target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
        )
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                joint_poses[i],
                maxVelocity=self.max_velocity,
            )

    def move_gripper(self, open_length):
        open_length = max(
            self.gripper_range[0], min(open_length, self.gripper_range[1])
        )
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(
            self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle
        )

    def get_current_ee_position(self):
        eef_state = p.getLinkState(self.id, self.eef_id)
        return eef_state

    def get_robot_state(self):
        """Get complete robot state: end-effector pose + joint angles"""
        # Get end-effector state
        eef_state = p.getLinkState(self.id, self.eef_id)
        eef_pos = np.array(eef_state[0])
        eef_orn_quat = np.array(eef_state[1])

        eef_orn_euler = np.array(p.getEulerFromQuaternion(eef_orn_quat))

        joint_states = []
        for joint_id in self.arm_controllable_joints:
            joint_state = p.getJointState(self.id, joint_id)
            joint_states.append(joint_state[0])  # Joint position
        """
        IMPORTANT : mimic_parent_name = 'finger_joint'
        This is the mimic_parent_id that controls the main gripper opening/closing. 
        In the code, this is set up in the __setup_mimic_joints__ method:So the gripper state saved in state[12] is the angle of joint ID 9 (finger_joint), 
        which ranges from 0.0 (closed) to 0.8 (open).
        The other gripper joints (IDs 11, 13, 14, 16, 18) are mimic joints that follow this parent 
        joint automatically through the gear constraints, so we only need to track this one joint to represent the full gripper state.
        """
        # Get gripper state
        gripper_state = p.getJointState(self.id, self.mimic_parent_id)
        gripper_angle = gripper_state[0]
        print(f"Gripper angle: {gripper_angle:.4f}")

        # Combine into single state vector
        # [eef_pos (3), eef_orn_euler (3), arm_joints (6), gripper (1)] = 13 dimensions
        state = np.concatenate(
            [
                eef_pos,  # 3D position (x, y, z)
                eef_orn_euler,  # 3D euler angles (roll, pitch, yaw)
                joint_states,  # 6 joint angles
                [gripper_angle],  # 1 gripper angle
            ]
        )

        return state


def create_data_folders(iter_folder):
    # Third person camera folders
    tp_rgb_dir = os.path.join(iter_folder, "third_person", "rgb")
    tp_depth_dir = os.path.join(iter_folder, "third_person", "depth")
    tp_pcd_dir = os.path.join(iter_folder, "third_person", "pcd")

    # Wrist camera folders
    wr_rgb_dir = os.path.join(iter_folder, "wrist", "rgb")
    wr_depth_dir = os.path.join(iter_folder, "wrist", "depth")
    wr_pcd_dir = os.path.join(iter_folder, "wrist", "pcd")

    # Camera poses folder
    poses_dir = os.path.join(iter_folder, "camera_poses")

    os.makedirs(tp_rgb_dir, exist_ok=True)
    os.makedirs(tp_depth_dir, exist_ok=True)
    os.makedirs(tp_pcd_dir, exist_ok=True)

    os.makedirs(wr_rgb_dir, exist_ok=True)
    os.makedirs(wr_depth_dir, exist_ok=True)
    os.makedirs(wr_pcd_dir, exist_ok=True)

    os.makedirs(poses_dir, exist_ok=True)

    return {
        "tp_rgb": tp_rgb_dir,
        "tp_depth": tp_depth_dir,
        "tp_pcd": tp_pcd_dir,
        "wr_rgb": wr_rgb_dir,
        "wr_depth": wr_depth_dir,
        "wr_pcd": wr_pcd_dir,
        "poses": poses_dir,
    }


def depth_to_point_cloud(depth_buffer, view_matrix, proj_matrix, width=224, height=224):
    """Convert depth buffer to 3D point cloud"""
    fov = 60
    near = 0.01
    far = 3.0

    # Convert depth buffer to actual depth values
    depth_img = far * near / (far - (far - near) * depth_buffer)

    # Create pixel grid
    fx = fy = width / (2 * np.tan(np.radians(fov) / 2))
    cx, cy = width / 2, height / 2

    # Get view matrix as numpy array
    view_matrix_np = np.array(view_matrix).reshape(4, 4).T

    # Create meshgrid for pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Convert to 3D coordinates in camera frame
    # Note: X is negated to correct for OpenGL/PyBullet coordinate convention
    # where the depth image produces a mirrored point cloud when visualized in Open3D
    z = depth_img
    x = -(u - cx) * z / fx
    y = -(v - cy) * z / fy

    # Stack into point cloud (N x 3)
    points_camera = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # # Transform to world coordinates
    # points_camera_homogeneous = np.concatenate([points_camera, np.ones((points_camera.shape[0], 1))], axis=1)
    # view_matrix_inv = np.linalg.inv(view_matrix_np)
    # points_world = (view_matrix_inv @ points_camera_homogeneous.T).T[:, :3]

    return points_camera


def get_wrist_camera_params(robot):
    """Get wrist camera position and orientation based on end-effector"""
    eef_state = robot.get_current_ee_position()
    eef_pos, eef_orn = eef_state[0], eef_state[1]

    # Convert quaternion to rotation matrix
    rot_matrix = np.array(p.getMatrixFromQuaternion(eef_orn)).reshape(3, 3)

    # Camera offset - further back and higher to capture gripper + cube + scene
    cam_offset_local = np.array([-0.05, 0, 0.12])  # Back 5cm, up 12cm
    cam_pos = np.array(eef_pos) + rot_matrix @ cam_offset_local

    # Camera target - look forward and down to see gripper, cube, and workspace
    cam_target = cam_pos + rot_matrix[:, 0] * 0.35 + rot_matrix[:, 2] * (-0.15)

    # Camera up vector
    cam_up = rot_matrix[:, 2]

    return cam_pos, cam_target, cam_up


def save_camera_pose(pose_dict, poses_dir, frame_idx):
    """Save camera pose information"""
    pose_file = os.path.join(poses_dir, f"pose_{frame_idx:04d}.json")
    with open(pose_file, "w") as f:
        json.dump(pose_dict, f, indent=2)


def compute_extrinsics(cam_pos, cam_target, cam_up):
    """Compute camera extrinsics matrix from position, target, and up vector"""
    # Camera coordinate system
    cam_pos = np.array(cam_pos)
    cam_target = np.array(cam_target)
    cam_up = np.array(cam_up)

    # Forward direction (z-axis in camera frame, points from camera to target)
    forward = cam_target - cam_pos
    forward = forward / np.linalg.norm(forward)

    # Right direction (x-axis in camera frame)
    right = np.cross(forward, cam_up)
    right = right / np.linalg.norm(right)

    # Recalculate up to ensure orthogonality (y-axis in camera frame)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Build rotation matrix (world to camera)
    # Rows are the camera axes expressed in world coordinates
    rotation_matrix = np.array(
        [
            right,
            up,
            -forward,  # Negative because camera looks down -z in OpenGL convention
        ]
    )

    # Translation vector (camera position in world frame)
    translation = cam_pos

    # Build 4x4 extrinsics matrix [R | t]
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = -rotation_matrix @ translation  # Camera to world transform

    return {
        "rotation_matrix": rotation_matrix.tolist(),
        "translation": translation.tolist(),
        "extrinsics_matrix": extrinsics.tolist(),
    }


def update_simulation(
    steps,
    sleep_time=0.01,
    capture_frames=False,
    iter_folder=None,
    frame_counter=None,
    robot=None,
    base_pos=None,
    state_history=None,
    cube_id=None,
    cube_pos_history=None,
    table_id=None,
    capture_table=False,
):
    """Update simulation and optionally capture frames from both cameras"""

    if capture_frames:
        dirs = create_data_folders(iter_folder)

    # Third-person camera (fixed)
    tp_cam_eye = [1.1, -0.6, 1.3]
    tp_cam_target = [0.5, 0.3, 0.7]
    tp_cam_up = [0, 0, 1]

    # Hide/show table during capture
    if table_id is not None and capture_frames:
        if capture_table:
            # Show table
            p.changeVisualShape(table_id, -1, rgbaColor=[1, 1, 1, 1])
        else:
            # Hide table by making it fully transparent
            p.changeVisualShape(table_id, -1, rgbaColor=[1, 1, 1, 0])

    for _ in range(steps):
        p.stepSimulation()

        if capture_frames and iter_folder is not None and frame_counter is not None:
            # ============ THIRD PERSON CAMERA ============
            view_matrix_tp = p.computeViewMatrix(
                cameraEyePosition=tp_cam_eye,
                cameraTargetPosition=tp_cam_target,
                cameraUpVector=tp_cam_up,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.01, farVal=3.0
            )

            # Capture third-person RGB and Depth
            width, height, rgb_tp, depth_tp, _ = p.getCameraImage(
                224, 224, viewMatrix=view_matrix_tp, projectionMatrix=proj_matrix
            )

            rgb_tp = np.array(rgb_tp)[:, :, :3]
            depth_buffer_tp = np.array(depth_tp)
            point_cloud_tp = depth_to_point_cloud(
                depth_buffer_tp, view_matrix_tp, proj_matrix
            )

            # Save third-person data
            cv2.imwrite(
                os.path.join(dirs["tp_rgb"], f"tp_rgb_{frame_counter[0]:04d}.png"),
                cv2.cvtColor(rgb_tp, cv2.COLOR_RGB2BGR),
            )
            np.save(
                os.path.join(dirs["tp_depth"], f"tp_depth_{frame_counter[0]:04d}.npy"),
                depth_buffer_tp,
            )
            np.save(
                os.path.join(dirs["tp_pcd"], f"tp_pcd_{frame_counter[0]:04d}.npy"),
                point_cloud_tp,
            )

            # Flatten RGB and point cloud
            colors_tp = rgb_tp.reshape(-1, 3)
            points_tp = point_cloud_tp.reshape(-1, 3)
            ply_path_tp = os.path.join(
                dirs["tp_pcd"], f"tp_pcd_{frame_counter[0]:04d}.ply"
            )
            save_point_cloud_ply(points_tp, colors_tp, ply_path_tp)

            # ============ WRIST CAMERA ============
            wr_cam_pos, wr_cam_target, wr_cam_up = get_wrist_camera_params(robot)

            view_matrix_wr = p.computeViewMatrix(
                cameraEyePosition=wr_cam_pos,
                cameraTargetPosition=wr_cam_target,
                cameraUpVector=wr_cam_up,
            )

            # Capture wrist RGB and Depth
            width, height, rgb_wr, depth_wr, _ = p.getCameraImage(
                224, 224, viewMatrix=view_matrix_wr, projectionMatrix=proj_matrix
            )

            rgb_wr = np.array(rgb_wr)[:, :, :3]
            depth_buffer_wr = np.array(depth_wr)
            point_cloud_wr = depth_to_point_cloud(
                depth_buffer_wr, view_matrix_wr, proj_matrix
            )

            # Save wrist data
            cv2.imwrite(
                os.path.join(dirs["wr_rgb"], f"wr_rgb_{frame_counter[0]:04d}.png"),
                cv2.cvtColor(rgb_wr, cv2.COLOR_RGB2BGR),
            )
            np.save(
                os.path.join(dirs["wr_depth"], f"wr_depth_{frame_counter[0]:04d}.npy"),
                depth_buffer_wr,
            )
            np.save(
                os.path.join(dirs["wr_pcd"], f"wr_pcd_{frame_counter[0]:04d}.npy"),
                point_cloud_wr,
            )

            colors_wr = rgb_wr.reshape(-1, 3)
            points_wr = point_cloud_wr.reshape(-1, 3)
            ply_path_wr = os.path.join(
                dirs["wr_pcd"], f"wr_pcd_{frame_counter[0]:04d}.ply"
            )
            save_point_cloud_ply(points_wr, colors_wr, ply_path_wr)

            # ============ SAVE CAMERA POSES ============
            # Calculate pose relative to robot base
            base_pos_array = np.array(base_pos)

            # Third-person camera extrinsics (constant, relative to base)
            tp_cam_pos_base = np.array(tp_cam_eye) - base_pos_array
            tp_cam_target_base = np.array(tp_cam_target) - base_pos_array
            tp_extrinsics = compute_extrinsics(
                tp_cam_pos_base, tp_cam_target_base, tp_cam_up
            )

            # Wrist camera extrinsics (changes every frame, relative to base)
            wr_cam_pos_base = np.array(wr_cam_pos) - base_pos_array
            wr_cam_target_base = np.array(wr_cam_target) - base_pos_array
            wr_extrinsics = compute_extrinsics(
                wr_cam_pos_base, wr_cam_target_base, wr_cam_up
            )

            pose_dict = {
                "frame": frame_counter[0],
                "third_person_camera": {
                    "rotation_matrix": tp_extrinsics["rotation_matrix"],
                    "translation": tp_extrinsics["translation"],
                    "extrinsics_matrix": tp_extrinsics["extrinsics_matrix"],
                },
                "wrist_camera": {
                    "rotation_matrix": wr_extrinsics["rotation_matrix"],
                    "translation": wr_extrinsics["translation"],
                    "extrinsics_matrix": wr_extrinsics["extrinsics_matrix"],
                },
            }

            save_camera_pose(pose_dict, dirs["poses"], frame_counter[0])

            # ============ SAVE ROBOT STATE ============
            current_state = robot.get_robot_state()
            if state_history is not None:
                state_history.append(current_state)

            # ============ SAVE CUBE POSITION ============
            if cube_id is not None and cube_pos_history is not None:
                cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
                # Store position and orientation (quaternion)
                cube_state = np.concatenate(
                    [
                        np.array(cube_pos),  # 3D position (x, y, z)
                        np.array(cube_orn),  # 4D quaternion (x, y, z, w)
                    ]
                )
                cube_pos_history.append(cube_state)

            frame_counter[0] += 1

    # Restore table visibility after capture
    if table_id is not None and capture_frames:
        p.changeVisualShape(table_id, -1, rgbaColor=[1, 1, 1, 1])


def save_point_cloud_ply(points, colors, filename):
    """Save point cloud in PLY format with colors"""
    valid_mask = points[:, 2] < 2.5
    points = points[valid_mask]
    colors = colors[valid_mask]

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for point, color in zip(points, colors):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} ")
            f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")


def setup_simulation(capture_table=False):
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if capture_table:
        p.loadURDF("plane.urdf", [0, 0, 0], useMaximalCoordinates=True)
    table_id = p.loadURDF(
        "table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0])
    )
    tray_pos = [0.5, 0.9, 0.6]
    tray_orn = p.getQuaternionFromEuler([0, 0, 0])
    p.loadURDF("tray/tray.urdf", tray_pos, tray_orn)
    return tray_pos, tray_orn, table_id


def random_color_cube(cube_id):
    color = [random.random(), random.random(), random.random(), 1.0]
    p.changeVisualShape(cube_id, -1, rgbaColor=color)


def move_and_grab_cube(
    robot, tray_pos, table_id, base_save_dir="dataset", capture_table=False
):
    iteration = 0
    while True:
        iter_folder = os.path.join(base_save_dir, f"iter_{iteration:04d}")
        os.makedirs(iter_folder, exist_ok=True)

        frame_counter = [0]
        state_history = []  # Store robot states for this iteration
        cube_pos_history = []  # Store cube positions for this iteration

        # Reset arm posture
        target_joint_positions = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            p.setJointMotorControl2(
                robot.id, joint_id, p.POSITION_CONTROL, target_joint_positions[i]
            )
        p.setJointMotorControl2(
            robot.id, 
            robot.mimic_parent_id,
            p.POSITION_CONTROL,
            targetPosition=0.0000,
            force=200
        )
        
            
        # Step simulation to stabilize
        for _ in range(5000):
            p.stepSimulation()

        
        update_simulation(
            200,
            capture_frames=False,
            iter_folder=iter_folder,
            frame_counter=frame_counter,
            robot=robot,
            base_pos=robot.base_pos,
            state_history=state_history,
            table_id=table_id,
            capture_table=capture_table,
        )

        # Random cube
        cube_start_pos = [random.uniform(0.3, 0.7), random.uniform(-0.1, 0.1), 0.65]
        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, cube_start_orn)
        random_color_cube(cube_id)

        # Get end-effector orientation
        eef_state = robot.get_current_ee_position()
        eef_orientation = eef_state[1]

        actual_gripper = p.getJointState(robot.id, robot.mimic_parent_id)[0]
        print(f"Reset complete - Gripper position: {actual_gripper:.4f} (should be ~0.8 for open)")


        # Move above cube
        robot.move_arm_ik([cube_start_pos[0], cube_start_pos[1], 0.83], eef_orientation)
        update_simulation(
            50,
            capture_frames=True,
            iter_folder=iter_folder,
            frame_counter=frame_counter,
            robot=robot,
            base_pos=robot.base_pos,
            state_history=state_history,
            cube_id=cube_id,
            cube_pos_history=cube_pos_history,
            table_id=table_id,
            capture_table=capture_table,
        )

        # Move down
        robot.move_arm_ik([cube_start_pos[0], cube_start_pos[1], 0.78], eef_orientation)
        update_simulation(
            50,
            capture_frames=True,
            iter_folder=iter_folder,
            frame_counter=frame_counter,
            robot=robot,
            base_pos=robot.base_pos,
            state_history=state_history,
            cube_id=cube_id,
            cube_pos_history=cube_pos_history,
            table_id=table_id,
            capture_table=capture_table,
        )

        # Close gripper
        robot.move_gripper(0.01)
        update_simulation(
            25,
            capture_frames=True,
            iter_folder=iter_folder,
            frame_counter=frame_counter,
            robot=robot,
            base_pos=robot.base_pos,
            state_history=state_history,
            cube_id=cube_id,
            cube_pos_history=cube_pos_history,
            table_id=table_id,
            capture_table=capture_table,
        )

        # Lift cube
        robot.move_arm_ik([cube_start_pos[0], cube_start_pos[1], 1.18], eef_orientation)
        update_simulation(
            50,
            capture_frames=True,
            iter_folder=iter_folder,
            frame_counter=frame_counter,
            robot=robot,
            base_pos=robot.base_pos,
            state_history=state_history,
            cube_id=cube_id,
            cube_pos_history=cube_pos_history,
            table_id=table_id,
            capture_table=capture_table,
        )

        # Move above tray
        tray_offset = random.uniform(0.1, 0.3)
        robot.move_arm_ik(
            [tray_pos[0] + tray_offset, tray_pos[1] + tray_offset, tray_pos[2] + 0.56],
            eef_orientation,
        )
        update_simulation(
            150,
            capture_frames=True,
            iter_folder=iter_folder,
            frame_counter=frame_counter,
            robot=robot,
            base_pos=robot.base_pos,
            state_history=state_history,
            cube_id=cube_id,
            cube_pos_history=cube_pos_history,
            table_id=table_id,
            capture_table=capture_table,
        )

        # Open gripper
        robot.move_gripper(0.085)
        update_simulation(
            25,
            capture_frames=True,
            iter_folder=iter_folder,
            frame_counter=frame_counter,
            robot=robot,
            base_pos=robot.base_pos,
            state_history=state_history,
            cube_id=cube_id,
            cube_pos_history=cube_pos_history,
            table_id=table_id,
            capture_table=capture_table,
        )

        # Remove cube
        p.removeBody(cube_id)

        # ============ SAVE AGENT STATES AND ACTIONS ============
        # Convert state history to numpy array
        agent_pos = np.array(state_history)  # Shape: (T, 13)

        # Compute actions as differences between consecutive states
        actions = np.diff(agent_pos, axis=0)  # Shape: (T-1, 13)
        # Pad with zeros for the last timestep to match length
        actions = np.vstack([actions, np.zeros(13)])  # Shape: (T, 13)

        # Convert cube position history to numpy array
        cube_positions = np.array(cube_pos_history)  # Shape: (T, 7) - 3 pos + 4 quat

        # Save to file
        state_action_data = {
            "agent_pos": agent_pos.tolist(),  # Robot states (T, 13)
            "action": actions.tolist(),  # Actions as state differences (T, 13)
            "cube_pos": cube_positions.tolist(),  # Cube positions (T, 7)
            "num_frames": len(agent_pos),
            "state_dim": 13,
            "cube_dim": 7,
            "state_description": {
                "eef_pos": [0, 1, 2],  # End-effector position (x, y, z)
                "eef_orn": [3, 4, 5],  # End-effector orientation (roll, pitch, yaw)
                "arm_joints": [6, 7, 8, 9, 10, 11],  # 6 arm joint angles
                "gripper": [12],  # Gripper angle
            },
            "cube_description": {
                "position": [0, 1, 2],  # Cube position (x, y, z)
                "orientation": [3, 4, 5, 6],  # Cube orientation quaternion (x, y, z, w)
            },
        }

        state_action_file = os.path.join(iter_folder, "state_action.json")
        with open(state_action_file, "w") as f:
            json.dump(state_action_data, f, indent=2)

        # Save as numpy arrays
        np.save(os.path.join(iter_folder, "agent_pos.npy"), agent_pos)
        np.save(os.path.join(iter_folder, "actions.npy"), actions)
        np.save(os.path.join(iter_folder, "cube_pos.npy"), cube_positions)

        # Save as text files
        np.savetxt(
            os.path.join(iter_folder, "agent_pos.txt"),
            agent_pos,
            fmt="%.6f",
            delimiter=" ",
            header="End-effector pose (x,y,z,roll,pitch,yaw) + 6 joint angles + gripper angle (13 dimensions)",
        )
        np.savetxt(
            os.path.join(iter_folder, "actions.txt"),
            actions,
            fmt="%.6f",
            delimiter=" ",
            header="Action deltas: differences between consecutive states (13 dimensions)",
        )
        np.savetxt(
            os.path.join(iter_folder, "cube_pos.txt"),
            cube_positions,
            fmt="%.6f",
            delimiter=" ",
            header="Cube position (x,y,z) + orientation quaternion (x,y,z,w) (7 dimensions)",
        )

        print(f"Completed iteration {iteration} - {frame_counter[0]} frames captured")
        print(f"  Agent states shape: {agent_pos.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Cube positions shape: {cube_positions.shape}")
        iteration += 1

        if(iteration >= 36):  # Limit to 1 iteration for testing
            break


def main():
    CAPTURE_TABLE = False  # Set to True to capture table in data
    tray_pos, tray_orn, table_id = setup_simulation(CAPTURE_TABLE)
    robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
    robot.load()
    # Set capture_table=False to hide table during data capture (default)
    # Set capture_table=True to show table during data capture
    move_and_grab_cube(robot, tray_pos, table_id, capture_table=CAPTURE_TABLE)


if __name__ == "__main__":
    main()

