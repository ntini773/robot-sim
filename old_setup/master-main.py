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

        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)

        print(f"Moving gripper to open length: {open_length:.4f} (angle: {open_angle:.4f})")
        p.setJointMotorControl2(
            self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle
        )

        print("Reached pos: ", p.getJointState(self.id, self.mimic_parent_id)[0])

    def get_current_ee_position(self):
        eef_state = p.getLinkState(self.id, self.eef_id)
        return eef_state

    def get_robot_state(self):
        """Get complete robot state: end-effector pose + joint angles"""
        eef_state = p.getLinkState(self.id, self.eef_id)
        eef_pos = np.array(eef_state[0])
        eef_orn_quat = np.array(eef_state[1])
        eef_orn_euler = np.array(p.getEulerFromQuaternion(eef_orn_quat))

        joint_states = []
        for joint_id in self.arm_controllable_joints:
            joint_state = p.getJointState(self.id, joint_id)
            joint_states.append(joint_state[0])

        gripper_state = p.getJointState(self.id, self.mimic_parent_id)
        gripper_angle = gripper_state[0]
        print(f"Gripper angle: {gripper_angle:.4f}")

        state = np.concatenate([eef_pos, eef_orn_euler, joint_states, [gripper_angle]])
        return state


def create_data_folders(iter_folder):
    tp_rgb_dir = os.path.join(iter_folder, "third_person", "rgb")
    tp_depth_dir = os.path.join(iter_folder, "third_person", "depth")
    tp_pcd_dir = os.path.join(iter_folder, "third_person", "pcd")
    tp_seg_dir = os.path.join(iter_folder, "third_person", "segmentation")

    wr_rgb_dir = os.path.join(iter_folder, "wrist", "rgb")
    wr_depth_dir = os.path.join(iter_folder, "wrist", "depth")
    wr_pcd_dir = os.path.join(iter_folder, "wrist", "pcd")
    wr_seg_dir = os.path.join(iter_folder, "wrist", "segmentation")

    poses_dir = os.path.join(iter_folder, "camera_poses")

    os.makedirs(tp_rgb_dir, exist_ok=True)
    os.makedirs(tp_depth_dir, exist_ok=True)
    os.makedirs(tp_pcd_dir, exist_ok=True)
    os.makedirs(tp_seg_dir, exist_ok=True)

    os.makedirs(wr_rgb_dir, exist_ok=True)
    os.makedirs(wr_depth_dir, exist_ok=True)
    os.makedirs(wr_pcd_dir, exist_ok=True)
    os.makedirs(wr_seg_dir, exist_ok=True)

    os.makedirs(poses_dir, exist_ok=True)

    return {
        "tp_rgb": tp_rgb_dir,
        "tp_depth": tp_depth_dir,
        "tp_pcd": tp_pcd_dir,
        "tp_seg": tp_seg_dir,
        "wr_rgb": wr_rgb_dir,
        "wr_depth": wr_depth_dir,
        "wr_pcd": wr_pcd_dir,
        "wr_seg": wr_seg_dir,
        "poses": poses_dir,
    }


def depth_to_point_cloud(depth_buffer, view_matrix, proj_matrix, width=224, height=224):
    """Convert depth buffer to 3D point cloud"""
    fov = 60
    near = 0.01
    far = 3.0

    depth_img = far * near / (far - (far - near) * depth_buffer)
    fx = fy = width / (2 * np.tan(np.radians(fov) / 2))
    cx, cy = width / 2, height / 2

    view_matrix_np = np.array(view_matrix).reshape(4, 4).T

    u, v = np.meshgrid(np.arange(width), np.arange(height))

    z = depth_img
    x = -(u - cx) * z / fx
    y = -(v - cy) * z / fy

    points_camera = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return points_camera


def get_wrist_camera_params(robot):
    """Get wrist camera position and orientation based on end-effector"""
    eef_state = robot.get_current_ee_position()
    eef_pos, eef_orn = eef_state[0], eef_state[1]

    rot_matrix = np.array(p.getMatrixFromQuaternion(eef_orn)).reshape(3, 3)
    cam_offset_local = np.array([-0.05, 0, 0.12])
    cam_pos = np.array(eef_pos) + rot_matrix @ cam_offset_local
    cam_target = cam_pos + rot_matrix[:, 0] * 0.35 + rot_matrix[:, 2] * (-0.15)
    cam_up = rot_matrix[:, 2]

    return cam_pos, cam_target, cam_up


def save_camera_pose(pose_dict, poses_dir, frame_idx):
    """Save camera pose information"""
    pose_file = os.path.join(poses_dir, f"pose_{frame_idx:04d}.json")
    with open(pose_file, "w") as f:
        json.dump(pose_dict, f, indent=2)


def compute_extrinsics(cam_pos, cam_target, cam_up):
    """Compute camera extrinsics matrix from position, target, and up vector"""
    cam_pos = np.array(cam_pos)
    cam_target = np.array(cam_target)
    cam_up = np.array(cam_up)

    forward = cam_target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, cam_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    rotation_matrix = np.array([right, up, -forward])
    translation = cam_pos

    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = -rotation_matrix @ translation

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
    plane_id=None,
    tray_id=None,
    EXCLUDE_TABLE = True,
):
    """Update simulation and capture frames with table segmentation"""

    if capture_frames:
        dirs = create_data_folders(iter_folder)

    tp_cam_eye = [1.1, -0.6, 1.3]
    tp_cam_target = [0.5, 0.3, 0.7]
    tp_cam_up = [0, 0, 1]

    # Create list of object IDs to exclude from point clouds
    exclude_ids = []
    if table_id is not None and EXCLUDE_TABLE:
        print("Excluding table from point clouds with {}".format(table_id))
        exclude_ids.append(table_id)
    if plane_id is not None:
        print("Excluding plane from point clouds with {}".format(plane_id))
        exclude_ids.append(plane_id)

    # if tray_id is not None:
    #     print("Excluding tray from point clouds with {}".format(tray_id))
    #     exclude_ids.append(tray_id)

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

            # Capture with segmentation mask
            width, height, rgb_tp, depth_tp, seg_tp = p.getCameraImage(
                224, 224, 
                viewMatrix=view_matrix_tp, 
                projectionMatrix=proj_matrix,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            )

            rgb_tp = np.array(rgb_tp)[:, :, :3]
            depth_buffer_tp = np.array(depth_tp)
            seg_tp = np.array(seg_tp)
            
            # Create mask for objects to exclude
            exclude_mask_tp = np.zeros_like(seg_tp, dtype=bool)
            for obj_id in exclude_ids:
                exclude_mask_tp |= (seg_tp == obj_id)
            
            point_cloud_tp = depth_to_point_cloud(
                depth_buffer_tp, view_matrix_tp, proj_matrix
            )

            # --- APPLY FILTERS TO NPY DATA ---
            points_tp_flat = point_cloud_tp.reshape(-1, 3)
            # Filter 1: Depth threshold
            valid_mask_tp = points_tp_flat[:, 2] < 2.5
            # Filter 2: Segmentation exclusion (Table/Plane)
            exclude_mask_flat_tp = exclude_mask_tp.flatten()
            final_mask_tp = valid_mask_tp & (~exclude_mask_flat_tp)
            
            # Filtered point cloud for NPY
            filtered_pcd_tp = points_tp_flat[final_mask_tp]

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
                filtered_pcd_tp, 
            )
            np.save(
                os.path.join(dirs["tp_seg"], f"tp_seg_{frame_counter[0]:04d}.npy"),
                seg_tp,
            )

            # Save PLY with table removed
            colors_tp = rgb_tp.reshape(-1, 3)
            points_tp = point_cloud_tp.reshape(-1, 3)
            exclude_mask_flat_tp = exclude_mask_tp.flatten()
            
            ply_path_tp = os.path.join(
                dirs["tp_pcd"], f"tp_pcd_{frame_counter[0]:04d}.ply"
            )
            save_point_cloud_ply(points_tp, colors_tp, ply_path_tp, exclude_mask_flat_tp)

            # ============ WRIST CAMERA ============
            wr_cam_pos, wr_cam_target, wr_cam_up = get_wrist_camera_params(robot)

            view_matrix_wr = p.computeViewMatrix(
                cameraEyePosition=wr_cam_pos,
                cameraTargetPosition=wr_cam_target,
                cameraUpVector=wr_cam_up,
            )

            # Capture with segmentation mask
            width, height, rgb_wr, depth_wr, seg_wr = p.getCameraImage(
                224, 224, 
                viewMatrix=view_matrix_wr, 
                projectionMatrix=proj_matrix,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            )

            rgb_wr = np.array(rgb_wr)[:, :, :3]
            depth_buffer_wr = np.array(depth_wr)
            seg_wr = np.array(seg_wr)
            
            # Create mask for objects to exclude
            exclude_mask_wr = np.zeros_like(seg_wr, dtype=bool)
            for obj_id in exclude_ids:
                exclude_mask_wr |= (seg_wr == obj_id)
            
            point_cloud_wr = depth_to_point_cloud(
                depth_buffer_wr, view_matrix_wr, proj_matrix
            )
            # --- APPLY FILTERS TO NPY DATA ---
            points_wr_flat = point_cloud_wr.reshape(-1, 3)
            # Filter 1: Depth threshold
            valid_mask_wr = points_wr_flat[:, 2] < 2.5
            # Filter 2: Segmentation exclusion
            exclude_mask_flat_wr = exclude_mask_wr.flatten()
            final_mask_wr = valid_mask_wr & (~exclude_mask_flat_wr)
            
            # Filtered point cloud for NPY
            filtered_pcd_wr = points_wr_flat[final_mask_wr]

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
                filtered_pcd_wr, # Saved filtered N x 3 array
            )
            np.save(
                os.path.join(dirs["wr_seg"], f"wr_seg_{frame_counter[0]:04d}.npy"),
                seg_wr,
            )

            # Save PLY with table removed
            colors_wr = rgb_wr.reshape(-1, 3)
            points_wr = point_cloud_wr.reshape(-1, 3)
            exclude_mask_flat_wr = exclude_mask_wr.flatten()
            
            ply_path_wr = os.path.join(
                dirs["wr_pcd"], f"wr_pcd_{frame_counter[0]:04d}.ply"
            )
            save_point_cloud_ply(points_wr, colors_wr, ply_path_wr, exclude_mask_flat_wr)

            # ============ SAVE CAMERA POSES ============
            base_pos_array = np.array(base_pos)

            tp_cam_pos_base = np.array(tp_cam_eye) - base_pos_array
            tp_cam_target_base = np.array(tp_cam_target) - base_pos_array
            tp_extrinsics = compute_extrinsics(
                tp_cam_pos_base, tp_cam_target_base, tp_cam_up
            )

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
                cube_state = np.concatenate([np.array(cube_pos), np.array(cube_orn)])
                cube_pos_history.append(cube_state)

            frame_counter[0] += 1


def save_point_cloud_ply(points, colors, filename, exclude_mask=None):
    """Save point cloud in PLY format with colors, excluding masked points"""
    # Apply depth filter
    valid_mask = points[:, 2] < 2.5
    
    # Apply exclude mask if provided
    if exclude_mask is not None:
        valid_mask = valid_mask & (~exclude_mask)
    
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


def setup_simulation():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useMaximalCoordinates=True)
    table_id = p.loadURDF(
        "table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0])
    )
    tray_pos = [0.5, 0.9, 0.6]
    tray_orn = p.getQuaternionFromEuler([0, 0, 0])
    tray_id = p.loadURDF("tray/tray.urdf", tray_pos, tray_orn)
    
    return tray_pos, tray_orn, table_id, plane_id, tray_id


def random_color_cube(cube_id):
    color = [random.random(), random.random(), random.random(), 1.0]
    p.changeVisualShape(cube_id, -1, rgbaColor=color)


def move_and_grab_cube(robot, tray_pos, table_id, plane_id, tray_id, EXCLUDE_TABLE, base_save_dir="dataset"):
    iteration = 0
    while True:
        iter_folder = os.path.join(base_save_dir, f"iter_{iteration:04d}")
        os.makedirs(iter_folder, exist_ok=True)

        frame_counter = [0]
        state_history = []
        cube_pos_history = []

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
            plane_id=plane_id,
            tray_id=tray_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE,
        )

        # Random cube
        cube_start_pos = [random.uniform(0.3, 0.7), random.uniform(-0.1, 0.1), 0.65]
        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, cube_start_orn)
        random_color_cube(cube_id)

        eef_state = robot.get_current_ee_position()
        eef_orientation = eef_state[1]

        actual_gripper = p.getJointState(robot.id, robot.mimic_parent_id)[0]
        print(f"Reset complete - Gripper position: {actual_gripper:.4f}")

        # Move above cube
        robot.move_arm_ik([cube_start_pos[0], cube_start_pos[1], 0.83], eef_orientation)
        update_simulation(
            50, capture_frames=True, iter_folder=iter_folder,
            frame_counter=frame_counter, robot=robot, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=tray_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )

        # Move down
        robot.move_arm_ik([cube_start_pos[0], cube_start_pos[1], 0.78], eef_orientation)
        update_simulation(
            50, capture_frames=True, iter_folder=iter_folder,
            frame_counter=frame_counter, robot=robot, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=tray_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )

        # Close gripper
        robot.move_gripper(0.01)
        update_simulation(
            25, capture_frames=True, iter_folder=iter_folder,
            frame_counter=frame_counter, robot=robot, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=tray_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )

        # Lift cube
        robot.move_arm_ik([cube_start_pos[0], cube_start_pos[1], 1.18], eef_orientation)
        update_simulation(
            50, capture_frames=True, iter_folder=iter_folder,
            frame_counter=frame_counter, robot=robot, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=tray_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )

        # Move above tray
        tray_offset = random.uniform(0.1, 0.3)
        robot.move_arm_ik(
            [tray_pos[0] + tray_offset, tray_pos[1] + tray_offset, tray_pos[2] + 0.56],
            eef_orientation,
        )
        update_simulation(
            150, capture_frames=True, iter_folder=iter_folder,
            frame_counter=frame_counter, robot=robot, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=tray_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )

        # Open gripper
        robot.move_gripper(0.085)
        update_simulation(
            25, capture_frames=True, iter_folder=iter_folder,
            frame_counter=frame_counter, robot=robot, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=tray_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )

        # Remove cube
        p.removeBody(cube_id)

        # ============ SAVE AGENT STATES AND ACTIONS ============
        agent_pos = np.array(state_history)
        actions = np.diff(agent_pos, axis=0)
        actions = np.vstack([actions, np.zeros(13)])
        cube_positions = np.array(cube_pos_history)

        state_action_data = {
            "agent_pos": agent_pos.tolist(),
            "action": actions.tolist(),
            "cube_pos": cube_positions.tolist(),
            "num_frames": len(agent_pos),
            "state_dim": 13,
            "cube_dim": 7,
            "state_description": {
                "eef_pos": [0, 1, 2],
                "eef_orn": [3, 4, 5],
                "arm_joints": [6, 7, 8, 9, 10, 11],
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
    EXCLUDE_TABLE = False
    tray_pos, tray_orn, table_id, plane_id, tray_id = setup_simulation()
    robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
    robot.load()
    # Set capture_table=False to hide table during data capture (default)
    # Set capture_table=True to show table during data capture
    move_and_grab_cube(robot, tray_pos, table_id, plane_id, tray_id , EXCLUDE_TABLE = EXCLUDE_TABLE)


if __name__ == "__main__":
    main()

