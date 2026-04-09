import os
import pybullet as p
import pybullet_data
import math
import time
import random
import shutil
from collections import namedtuple
import cv2
import numpy as np
import json
import fpsample

import ompl.base as ob
import ompl.geometric as og
from planner_with_collision_robot import RobotOMPLPlanner, solve_ik_collision_free, solve_ik, visualise_eef_traj

CUBE_LENGTH = 0.05


# =============================================================================
# FRANKA ROBOT
# =============================================================================

class FrankaRobot:
    """
    Franka Panda robot — same interface as Lite6Robot.
    7-DOF arm + 2-finger parallel gripper (joints 9 & 10 in panda.urdf).

    Gripper convention (matches Franka real robot):
        0.04  = fully open  (4 cm each finger)
        0.0   = fully closed
    """

    URDF_PATH = "franka_panda/panda.urdf"

    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        # panda_hand  -> link 11 (index from 0),  panda_grasptarget -> link 11 as well
        # Use link 11 (panda_hand) as EEF — can be overridden in main() to use grasptarget
        self.eef_id       = 11
        self.arm_num_dofs = 7
        self.arm_rest_poses = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.max_velocity = 2.0
        # Gripper: open=0.04, closed=0.0
        self.gripper_open_val   =  0.04
        self.gripper_closed_val =  0.0
        self.gripper_grasp_val  =  0.016   # good for 5 cm cube (half-width 0.025 - margin)
        self.id = None

    def load(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.id = p.loadURDF(
            self.URDF_PATH,
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
        )
        self._parse_joint_info()
        self._setup_gripper()
        self._print_debug_info()

    def _parse_joint_info(self):
        JointInfo = namedtuple(
            "JointInfo",
            ["id", "name", "type", "lowerLimit", "upperLimit",
             "maxForce", "maxVelocity", "controllable"],
        )
        self.joints = []
        self.controllable_joints = []

        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jid   = info[0]
            jname = info[1].decode("utf-8")
            jtype = info[2]
            jll   = info[8]
            jul   = info[9]
            jmf   = info[10]
            jmv   = info[11]
            ctrl  = jtype != p.JOINT_FIXED

            if ctrl:
                self.controllable_joints.append(jid)

            self.joints.append(
                JointInfo(jid, jname, jtype, jll, jul, jmf, jmv, ctrl)
            )

        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits  = [j.lowerLimit  for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits  = [j.upperLimit  for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges  = [
            ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)
        ]

    def _setup_gripper(self):
        """
        Identify the two finger joints and create a gear constraint so both
        fingers mirror each other (standard Franka behaviour).
        panda.urdf finger joints:  panda_finger_joint1 (left), panda_finger_joint2 (right)
        """
        finger_names = ["panda_finger_joint1", "panda_finger_joint2"]
        self.finger_joint_ids = [
            j.id for j in self.joints if j.name in finger_names
        ]
        if len(self.finger_joint_ids) < 2:
            raise RuntimeError(
                f"Could not find finger joints {finger_names}. "
                f"Found: {[j.name for j in self.joints if j.controllable]}"
            )

        # finger1 is the "parent" we control; finger2 mirrors it
        self.mimic_parent_id       = self.finger_joint_ids[0]
        self.mimic_child_multiplier = {self.finger_joint_ids[1]: 1}

        c = p.createConstraint(
            self.id, self.mimic_parent_id,
            self.id, self.finger_joint_ids[1],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=-1, maxForce=100000, erp=1)

        # High friction on finger pads
        for fid in self.finger_joint_ids:
            p.changeDynamics(self.id, fid,
                             lateralFriction=5.0,
                             spinningFriction=5.0,
                             rollingFriction=5.0)

    def _print_debug_info(self):
        print("\n" + "=" * 60)
        print("FRANKA ROBOT DEBUG INFO")
        print("=" * 60)
        jtype_names = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED"}
        for j in self.joints:
            jtype = jtype_names.get(j.type, str(j.type))
            ctrl  = "CTRL" if j.controllable else "    "
            print(f"  Joint {j.id:2d}: {j.name:<40s} ({jtype:<9s}) [{ctrl}]")

        print(f"\nArm controllable joints : {self.arm_controllable_joints}")
        print(f"Arm lower limits        : {self.arm_lower_limits}")
        print(f"Arm upper limits        : {self.arm_upper_limits}")
        print(f"EEF link index          : {self.eef_id}")
        print(f"Finger joint IDs        : {self.finger_joint_ids}")
        eef_state = p.getLinkState(self.id, self.eef_id)
        print(f"EEF position            : {eef_state[0]}")
        print(f"EEF orientation (quat)  : {eef_state[1]}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move_arm_ik(self, target_pos, target_orn):
        joint_poses = p.calculateInverseKinematics(
            self.id, self.eef_id,
            target_pos, target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id, joint_id,
                p.POSITION_CONTROL,
                joint_poses[i],
                maxVelocity=self.max_velocity,
            )

    def move_gripper(self, open_val):
        """
        Directly set finger joints.
        open_val: 0.04 = open, 0.0 = closed, 0.016 = grasp 5 cm cube
        """
        p.resetJointState(self.id, self.mimic_parent_id, open_val)
        p.resetJointState(self.id, self.finger_joint_ids[1], open_val)
        for _ in range(10):
            p.stepSimulation()

    def reset_posture(self):
        """Teleport to rest pose with open gripper, then settle."""
        # Home pose: arm above table, pointing down
        initial_pos_world = [0.4, 0.0, 0.7]
        initial_orn_world = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])

        # Disable all motors before teleport
        for j in range(p.getNumJoints(self.id)):
            p.setJointMotorControl2(self.id, j, p.VELOCITY_CONTROL, force=0)

        # Seed IK with rest poses
        for i, jid in enumerate(self.arm_controllable_joints):
            p.resetJointState(self.id, jid, self.arm_rest_poses[i], targetVelocity=0)

        target_joints = p.calculateInverseKinematics(
            self.id, self.eef_id,
            initial_pos_world, initial_orn_world,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
        )

        for i, jid in enumerate(self.arm_controllable_joints):
            p.resetJointState(self.id, jid, target_joints[i], targetVelocity=0)
            p.setJointMotorControl2(
                self.id, jid, p.POSITION_CONTROL,
                targetPosition=target_joints[i], force=1000,
            )

        # Open gripper
        for fid in self.finger_joint_ids:
            p.resetJointState(self.id, fid, self.gripper_open_val, targetVelocity=0)
            p.setJointMotorControl2(
                self.id, fid, p.POSITION_CONTROL,
                targetPosition=self.gripper_open_val, force=500,
            )

        for _ in range(100):
            p.stepSimulation()

    def get_current_ee_position(self):
        eef_state = p.getLinkState(self.id, self.eef_id)
        return eef_state[0], eef_state[1]

    def get_robot_state(self):
        """
        Returns 8-dim state: 7 arm joints + 1 normalised gripper [0=open, 1=closed].
        """
        joint_states = [p.getJointState(self.id, j)[0]
                        for j in self.arm_controllable_joints]

        raw = p.getJointState(self.id, self.mimic_parent_id)[0]
        # open=0.04 -> 0.0,  closed=0.0 -> 1.0
        norm = (self.gripper_open_val - raw) / (self.gripper_open_val - self.gripper_closed_val + 1e-8)
        norm = float(np.clip(norm, 0.0, 1.0))

        return np.concatenate([joint_states, [norm]])


# =============================================================================
# GRIPPER INTERPOLATION  (Franka-specific)
# =============================================================================

def interpolate_gripper(robot, target_angle,
                        capture_frames=True, iter_folder=None,
                        frame_counter=None, base_pos=None,
                        state_history=None, cube_id=None,
                        cube_pos_history=None, table_id=None,
                        plane_id=None, tray_id=None, EXCLUDE_TABLE=True):
    """
    Franka gripper:
        target_angle < 0.01  ->  OPEN  (0.04 m each finger)
        target_angle >= 0.01 ->  CLOSE (0.0  m = contact, or stops at grasp threshold)
    """
    if target_angle < 0.01:
        finger_target = robot.gripper_open_val      # 0.04
    else:
        finger_target = robot.gripper_closed_val    # 0.0 — let physics stop at cube

    max_iters = 150
    for _ in range(max_iters):
        for fid in robot.finger_joint_ids:
            p.setJointMotorControl2(
                robot.id, fid,
                p.POSITION_CONTROL,
                targetPosition=finger_target,
                force=200,
            )

        update_simulation(
            1,
            capture_frames=capture_frames,
            iter_folder=iter_folder,
            frame_counter=frame_counter,
            robot=robot,
            base_pos=base_pos,
            state_history=state_history,
            cube_id=cube_id,
            cube_pos_history=cube_pos_history,
            table_id=table_id,
            plane_id=plane_id,
            tray_id=tray_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE,
        )

        current = p.getJointState(robot.id, robot.mimic_parent_id)[0]

        if finger_target < 0.01:   # closing — stop when fingers contact cube
            if current < robot.gripper_grasp_val:
                break
        else:                       # opening — stop when sufficiently open
            if current > 0.038:
                break

    final = p.getJointState(robot.id, robot.mimic_parent_id)[0]
    print(f"Final gripper position: {final:.4f}  (target={finger_target:.4f})")


# =============================================================================
# UTILITIES  (unchanged from Lite6 version)
# =============================================================================

def create_cylinder(radius, height, pos, color=[0.7, 0.7, 0.7, 1]):
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[pos[0], pos[1], pos[2] + height / 2],
    )


def create_obstacle_box(half_extents, pos, color=[0.8, 0.2, 0.2, 1]):
    """Create a static solid-box obstacle sitting on the table."""
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
    )


def create_data_folders(iter_folder):
    dirs = {
        "tp_rgb":   os.path.join(iter_folder, "third_person", "rgb"),
        "tp_depth": os.path.join(iter_folder, "third_person", "depth"),
        "tp_pcd":   os.path.join(iter_folder, "third_person", "pcd"),
        "tp_seg":   os.path.join(iter_folder, "third_person", "segmentation"),
        "poses":    os.path.join(iter_folder, "camera_poses"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def depth_to_point_cloud(depth_buffer, view_matrix, proj_matrix, base_pos,
                          width=224, height=224):
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)
    x_ndc = (2.0 * u / width) - 1.0
    y_ndc = 1.0 - (2.0 * v / height)
    z_ndc = 2.0 * depth_buffer - 1.0
    ndc = np.stack([x_ndc, y_ndc, z_ndc, np.ones_like(z_ndc)], axis=-1).reshape(-1, 4)
    view_np = np.array(view_matrix).reshape(4, 4).T
    proj_np = np.array(proj_matrix).reshape(4, 4).T
    inv_vp = np.linalg.inv(proj_np @ view_np)
    world_homo = (inv_vp @ ndc.T).T
    points_world = world_homo[:, :3] / world_homo[:, 3:4]
    return points_world - np.array(base_pos)


def farthest_point_sampling(points, n_samples, colors=None):
    if len(points) <= n_samples:
        return (points, colors) if colors is not None else (points, None)
    indices = fpsample.fps_npdu_sampling(points, n_samples)
    return points[indices], (colors[indices] if colors is not None else None)


def save_camera_pose(pose_dict, poses_dir, frame_idx):
    with open(os.path.join(poses_dir, f"pose_{frame_idx:04d}.json"), "w") as f:
        json.dump(pose_dict, f, indent=2)


def compute_extrinsics(cam_pos, cam_target, cam_up):
    cam_pos    = np.array(cam_pos)
    cam_target = np.array(cam_target)
    cam_up     = np.array(cam_up)
    fwd   = cam_target - cam_pos;  fwd  /= np.linalg.norm(fwd)
    right = np.cross(fwd, cam_up); right /= np.linalg.norm(right)
    up    = np.cross(right, fwd)
    R = np.array([right, up, -fwd])
    E = np.eye(4); E[:3, :3] = R; E[:3, 3] = -R @ cam_pos
    return {"rotation_matrix": R.tolist(), "translation": cam_pos.tolist(),
            "extrinsics_matrix": E.tolist()}


def save_point_cloud_ply(points, colors, filename, exclude_mask=None):
    valid = points[:, 2] < 2.5
    if exclude_mask is not None:
        valid = valid & (~exclude_mask)
    points = points[valid]; colors = colors[valid]
    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        for prop in ["x", "y", "z"]:
            f.write(f"property float {prop}\n")
        for prop in ["red", "green", "blue"]:
            f.write(f"property uchar {prop}\n")
        f.write("end_header\n")
        for pt, cl in zip(points, colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{int(cl[0])} {int(cl[1])} {int(cl[2])}\n")


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
    EXCLUDE_TABLE=True,
):
    if capture_frames:
        dirs = create_data_folders(iter_folder)

    # Camera calibrated for Franka base at z=0 (table surface ~0.625)
    tp_cam_eye    = [0.7463, 0.3093, 0.5574]
    tp_cam_target = [0.7463 - 0.7751, 0.3093 - 0.4045, 0.5574 - 0.4855]
    tp_cam_up     = [0, 0, 1]

    exclude_ids = []
    if table_id is not None and EXCLUDE_TABLE:
        exclude_ids.append(table_id)
    if plane_id is not None:
        exclude_ids.append(plane_id)

    for _ in range(steps):
        p.stepSimulation()

        if capture_frames and iter_folder is not None and frame_counter is not None:
            view_matrix_tp = p.computeViewMatrix(tp_cam_eye, tp_cam_target, tp_cam_up)
            proj_matrix    = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.01, farVal=3.0
            )

            width, height, rgb_tp, depth_tp, seg_tp = p.getCameraImage(
                224, 224,
                viewMatrix=view_matrix_tp,
                projectionMatrix=proj_matrix,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            )

            rgb_tp         = np.array(rgb_tp)[:, :, :3]
            depth_buffer   = np.array(depth_tp)
            seg_tp         = np.array(seg_tp)

            exclude_mask = np.zeros_like(seg_tp, dtype=bool)
            for oid in exclude_ids:
                exclude_mask |= (seg_tp == oid)

            bg_mask = (depth_buffer >= 0.9999).flatten()

            pcd = depth_to_point_cloud(depth_buffer, view_matrix_tp, proj_matrix, base_pos)
            pts_flat = pcd.reshape(-1, 3)

            valid   = pts_flat[:, 2] < 2.5
            ex_flat = exclude_mask.flatten()
            keep    = valid & (~ex_flat) & (~bg_mask)
            filtered_pcd, _ = farthest_point_sampling(pts_flat[keep], 2500)

            # Save
            fc = frame_counter[0]
            cv2.imwrite(os.path.join(dirs["tp_rgb"],   f"tp_rgb_{fc:04d}.png"),
                        cv2.cvtColor(rgb_tp, cv2.COLOR_RGB2BGR))
            np.save(os.path.join(dirs["tp_depth"], f"tp_depth_{fc:04d}.npy"), depth_buffer)
            np.save(os.path.join(dirs["tp_pcd"],   f"tp_pcd_{fc:04d}.npy"),   filtered_pcd)
            np.save(os.path.join(dirs["tp_seg"],   f"tp_seg_{fc:04d}.npy"),   seg_tp)

            # PLY with FPS
            colors_all = rgb_tp.reshape(-1, 3)
            pts_all    = pcd.reshape(-1, 3)
            ply_keep   = (~(exclude_mask.flatten() | bg_mask)) & (pts_all[:, 2] < 2.5)
            ply_pts, ply_col = farthest_point_sampling(
                pts_all[ply_keep], 2500, colors_all[ply_keep]
            )
            save_point_cloud_ply(ply_pts, ply_col,
                                 os.path.join(dirs["tp_pcd"], f"tp_pcd_{fc:04d}.ply"))

            # Camera pose
            bp = np.array(base_pos)
            tp_ext = compute_extrinsics(
                np.array(tp_cam_eye) - bp,
                np.array(tp_cam_target) - bp,
                tp_cam_up,
            )
            save_camera_pose(
                {"frame": fc, "third_person_camera": tp_ext},
                dirs["poses"], fc,
            )

            # Robot + cube state
            if state_history is not None:
                state_history.append(robot.get_robot_state())

            if cube_id is not None and cube_pos_history is not None:
                cpos, corn = p.getBasePositionAndOrientation(cube_id)
                cube_pos_history.append(np.concatenate([np.array(cpos), np.array(corn)]))

            frame_counter[0] += 1


# =============================================================================
# MOTION HELPERS
# =============================================================================

def best_down_orn(robot, target_pos, n_yaws=4, obstacle_xys=None):
    """
    Return the gripper-down quaternion (roll=π, pitch=0, yaw=free) that:

    1. Has a valid IK solution within joint limits.
    2. Minimises joint-space travel from the current arm configuration.
    3. (When obstacle_xys provided) Orients the finger-spread axis toward
       the most laterally open direction around target_pos.

    For Franka panda the fingers open along the hand's local-X axis, which
    maps to world direction [cos(yaw), sin(yaw)] when roll=π, pitch=0.
    We therefore prefer the yaw whose finger axis is perpendicular to the
    dominant obstacle direction — i.e. the cube's 'open' side.
    """
    current_q = np.array([p.getJointState(robot.id, j)[0]
                           for j in robot.arm_controllable_joints])
    yaw_angles = np.linspace(0, 2 * math.pi, n_yaws, endpoint=False)

    lower = np.array(robot.arm_lower_limits)
    upper = np.array(robot.arm_upper_limits)
    joint_range_sum = float(np.sum(upper - lower)) + 1e-8

    # ── Compute the 2-D "free" direction around target_pos ──────────────────
    # Each nearby obstacle pushes a repulsion vector toward the cube.
    # The free finger-spread axis is PERPENDICULAR to the net repulsion,
    # i.e. along the axis the cube is *not* blocked on.
    free_dir = np.array([1.0, 0.0])          # default: align fingers with world-X
    if obstacle_xys and len(obstacle_xys) > 0:
        txy = np.array(target_pos[:2])
        repulsion = np.zeros(2)
        for oxy in obstacle_xys:
            diff = txy - np.array(oxy[:2])
            d = np.linalg.norm(diff)
            if d > 1e-3:
                repulsion += diff / (d * d)   # weight closer obstacles more
        norm_rep = np.linalg.norm(repulsion)
        if norm_rep > 1e-3:
            dom = repulsion / norm_rep        # dominant obstacle direction
            # free axis = perpendicular to dominant obstacle direction
            free_dir = np.array([-dom[1], dom[0]])

    best_orn  = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])  # fallback
    best_score = float("inf")

    for yaw in yaw_angles:
        orn = p.getQuaternionFromEuler([math.pi, 0.0, yaw])
        q = p.calculateInverseKinematics(
            robot.id, robot.eef_id,
            target_pos, orn,
            lowerLimits=robot.arm_lower_limits,
            upperLimits=robot.arm_upper_limits,
            jointRanges=robot.arm_joint_ranges,
            restPoses=robot.arm_rest_poses,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        q = np.array(q[:robot.arm_num_dofs])
        if np.any(q < lower - 1e-3) or np.any(q > upper + 1e-3):
            continue

        # ── Joint-travel term (0 = current pose, 1 = full joint range away) ─
        joint_term = np.linalg.norm(q - current_q) / joint_range_sum

        # ── Clearance term: how well the finger axis aligns with free_dir ───
        # Panda fingers spread along hand local-X → world [cos(yaw), sin(yaw)]
        finger_dir = np.array([math.cos(yaw), math.sin(yaw)])
        # |dot| because spreading in +free or -free is both fine
        alignment = abs(float(np.dot(finger_dir, free_dir)))
        clearance_term = 1.0 - alignment          # 0 = perfectly aligned

        # Combined score — lower is better.
        # Weight clearance higher so it dominates when cube is near obstacles.
        score = joint_term + 1.5 * clearance_term

        if score < best_score:
            best_score = score
            best_orn   = orn

    return best_orn


def move_to_pose_dynamic(robot, target_pos, target_orn,
                          max_steps=200, capture_frames=False,
                          iter_folder=None, frame_counter=None,
                          threshold=0.01, **kwargs):
    for step in range(max_steps):
        robot.move_arm_ik(target_pos, target_orn)
        update_simulation(1, robot=robot, capture_frames=capture_frames,
                          iter_folder=iter_folder, frame_counter=frame_counter,
                          **kwargs)
        cur_pos, _ = robot.get_current_ee_position()
        if np.linalg.norm(np.array(cur_pos) - np.array(target_pos)) < threshold:
            return True
    print(f"  -> Warning: Did not reach target in {max_steps} steps.")
    return False


def move_with_planner(planner, robot, target_pos, target_orn,
                      max_steps_per_waypoint=10, capture_frames=False,
                      iter_folder=None, frame_counter=None,
                      threshold=0.01, **kwargs):
    start_config = robot.get_robot_state()[:robot.arm_num_dofs]
    goal_config  = solve_ik(robot, planner, target_pos, target_orn)
    if goal_config is None:
        print("✗ Planning failed — IK could not find a solution.")
        return False

    planner._snapshot_gripper_pose()
    val = p.getJointState(robot.id, robot.mimic_parent_id)[0]
    success, path = planner.plan(start_config, goal_config, planning_time=10.0)
    planner._apply_frozen_gripper_printing()

    if not path:
        print("✗ Planning failed — no path found.")
        return False

    # Reset arm to start config (planning may have moved ghost/real robot)
    for i, jid in enumerate(robot.arm_controllable_joints):
        p.resetJointState(robot.id, jid, start_config[i])

    print(f"\nExecuting path with {len(path)} waypoints...")
    for i, waypoint in enumerate(path):
        for j, jid in enumerate(planner.joint_ids):
            p.setJointMotorControl2(
                robot.id, jid,
                p.POSITION_CONTROL,
                waypoint[j],
                maxVelocity=robot.max_velocity,
                force=200,
            )

        eef_state = p.getLinkState(robot.id, robot.eef_id)
        visualise_eef_traj(eef_state[0])

        for _ in range(max_steps_per_waypoint):
            print("GRIPPER DEBUG INFO:{}".format(p.getJointState(robot.id, robot.mimic_parent_id)[0]))
            # Hold gripper throughout motion
            for fid in robot.finger_joint_ids:
                p.setJointMotorControl2(
                    robot.id, fid, p.POSITION_CONTROL,
                    targetPosition=val,
                    force=500,
                )
            update_simulation(1, robot=robot, capture_frames=capture_frames,
                              iter_folder=iter_folder, frame_counter=frame_counter,
                              **kwargs)
            current = [p.getJointState(robot.id, j)[0] for j in planner.joint_ids]
            if np.max(np.abs(np.array(current) - np.array(waypoint))) < 0.01:
                break

        if i % max(1, len(path) // 10) == 0:
            print(f"  Progress: {i+1}/{len(path)} waypoints")

    return True


# =============================================================================
# SIMULATION SETUP
# =============================================================================

def setup_simulation(freq=60, gui=False):
    if gui:
        p.connect(p.GUI)
        print("PyBullet running in GUI mode")
    else:
        p.connect(p.DIRECT)
        print("PyBullet running in DIRECT (headless) mode")

    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(1 / freq)

    plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useMaximalCoordinates=True)
    table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0],
                           p.getQuaternionFromEuler([0, 0, 0]))
    return None, table_id, plane_id


# =============================================================================
# MAIN DATA COLLECTION LOOP
# =============================================================================

def move_and_grab_cube(robot, table_id, plane_id,
                        EXCLUDE_TABLE=True, base_save_dir="dataset_franka"):
    successful_iterations = 0
    total_attempts        = 0
    failed_attempts       = 0

    print("\n" + "=" * 60)
    print("STARTING FRANKA DATA COLLECTION")
    print("Target: 150 successful trajectories")
    print("=" * 60 + "\n")

    robot.reset_posture()

    cube_id     = None
    cylinder_id = None
    obstacle_ids = []  # two static box obstacles per iteration

    # Fixed obstacle geometry: 8 cm × 8 cm base, 12 cm tall
    # Kept moderate to avoid collisions at the robot's start configuration.
    OBS_HALF = [0.04, 0.04, 0.1]
    TABLE_Z  = 0.625  # table surface height
    OBS_COLORS = [
        [0.85, 0.20, 0.20, 1.0],   # red
        [0.20, 0.20, 0.85, 1.0],   # blue
    ]
    HOME_XY = np.array([0.40, 0.00])
    MIN_HOME_CLEARANCE = 0.22
    MIN_OBS_PAIR_DIST = 0.18

    while successful_iterations < 150:
        temp_folder = os.path.join(base_save_dir, f"temp_iter_{total_attempts:04d}")
        os.makedirs(temp_folder, exist_ok=True)

        # Clean up previous bodies
        for bid in [cube_id, cylinder_id] + obstacle_ids:
            if bid is not None:
                try:
                    p.removeBody(bid)
                except Exception:
                    pass
        cube_id = cylinder_id = None
        obstacle_ids = []

        # ------------------------------------------------------------------
        # Spawn 2 random box obstacles on the table (act as path blockers)
        # Keep them clear of the robot home/start region to avoid
        # "start configuration is in collision" planning failures.
        # ------------------------------------------------------------------
        def sample_obstacle_xy(x_rng, y_rng, existing_xy):
            for _ in range(40):
                cand = np.array([
                    random.uniform(*x_rng),
                    random.uniform(*y_rng),
                ])
                if np.linalg.norm(cand - HOME_XY) < MIN_HOME_CLEARANCE:
                    continue
                if any(np.linalg.norm(cand - prev) < MIN_OBS_PAIR_DIST for prev in existing_xy):
                    continue
                return cand
            return np.array([np.mean(x_rng), np.mean(y_rng)])

        obs_xy = []
        obs_xy.append(sample_obstacle_xy((0.48, 0.75), (-0.26, -0.12), obs_xy))
        obs_xy.append(sample_obstacle_xy((0.48, 0.78), (0.12, 0.26), obs_xy))

        obs_positions = [
            [xy[0], xy[1], TABLE_Z + OBS_HALF[2]]
            for xy in obs_xy
        ]
        for i, (obs_pos, obs_col) in enumerate(zip(obs_positions, OBS_COLORS)):
            oid = create_obstacle_box(OBS_HALF, obs_pos, color=obs_col)
            obstacle_ids.append(oid)
            print(f"Obstacle {i+1} at: {[f'{v:.4f}' for v in obs_pos]}")

        # Spawn target cylinder
        target_radius = 0.05
        target_height = 0.04
        tray_pos = [
            random.uniform(0.25, 0.8),
            random.uniform(-0.25, 0.25),
            0.625,
        ]
        cylinder_id = create_cylinder(target_radius, target_height, tray_pos,
                                       color=[1, 0.5, 0, 1])
        print(f"Cylinder at: {[f'{v:.4f}' for v in tray_pos]}")

        frame_counter     = [0]
        state_history     = []
        cube_pos_history  = []

        print(f"\n{'='*60}")
        print(f"ATTEMPT {total_attempts+1}  (Successful: {successful_iterations}/150)")
        print(f"{'='*60}\n")

        robot.reset_posture()
        print(f"Robot reset. Gripper: {p.getJointState(robot.id, robot.mimic_parent_id)[0]:.4f}\n")

        # Spawn cube
        cube_start_pos = [
            random.uniform(0.20, 0.80),
            random.uniform(-0.22, 0.12),
            0.65,
        ]
        cube_id = p.loadURDF("cube_small.urdf", cube_start_pos,
                              p.getQuaternionFromEuler([0, 0, 0]))
        p.changeDynamics(cube_id, -1, lateralFriction=5.0,
                          spinningFriction=5.0, rollingFriction=5.0)
        p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0, 0, 1])
        print(f"Cube at: {[f'{v:.4f}' for v in cube_start_pos]}")

        # ------------------------------------------------------------------
        # Build planners (two instances, different obstacle sets)
        # planner1: approach cube  — cube in obstacles so arm doesn't hit it
        # planner2: carry to tray  — cylinder in obstacles so arm doesn't hit it
        # ------------------------------------------------------------------
        planner1 = RobotOMPLPlanner(
            robot=robot,
            robot_urdf="franka_panda/panda.urdf",
            obstacles=[table_id, cube_id] + obstacle_ids,
            collision_margin=0.02,
            ignore_base=True,
        )
        planner1.set_planner("AITstar")

        planner2 = RobotOMPLPlanner(
            robot=robot,
            robot_urdf="franka_panda/panda.urdf",
            obstacles=[table_id, cylinder_id] + obstacle_ids,
            collision_margin=0.02,
            ignore_base=True,
        )
        planner2.set_planner("AITstar")

        common_kwargs = dict(
            capture_frames=True, iter_folder=temp_folder,
            frame_counter=frame_counter, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=cylinder_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE,
        )

        # XY centres of static obstacles — used for clearance-aware wrist yaw
        obs_xys = [op[:2] for op in obs_positions]

        # ------------------------------------------------------------------
        # Phase 1: Move above cube (planner — avoids cube)
        # ------------------------------------------------------------------
        print("Phase 1: Moving above cube...")
        above_cube = [cube_start_pos[0], cube_start_pos[1], 0.80]
        orn_above_cube = best_down_orn(robot, above_cube, obstacle_xys=obs_xys)
        move_with_planner(
            planner1, robot,
            above_cube, orn_above_cube,
            **common_kwargs,
        )
        step1_history = state_history.copy()

        # ------------------------------------------------------------------
        # Phase 2: Move down to grasp (closed-loop IK, no planner)
        # ------------------------------------------------------------------
        print("Phase 2: Moving down to grasp...")
        grasp_pos = [cube_start_pos[0], cube_start_pos[1], 0.645]
        orn_grasp = best_down_orn(robot, grasp_pos, obstacle_xys=obs_xys)
        move_to_pose_dynamic(
            robot,
            grasp_pos, orn_grasp,
            **common_kwargs,
        )
        planner1.cleanup()
        step2_history = state_history.copy()

        # ------------------------------------------------------------------
        # Phase 3: Close gripper
        # ------------------------------------------------------------------
        print("Phase 3: Closing gripper...")
        interpolate_gripper(robot, target_angle=0.5, **common_kwargs)
        step3_history = state_history.copy()

        # ------------------------------------------------------------------
        # Phase 4: Lift cube
        # ------------------------------------------------------------------
        print("Phase 4: Lifting cube...")
        lift_pos = [cube_start_pos[0], cube_start_pos[1], 0.80]
        orn_lift = best_down_orn(robot, lift_pos, obstacle_xys=obs_xys)
        move_to_pose_dynamic(
            robot,
            lift_pos, orn_lift,
            **common_kwargs,
        )
        step4_history = state_history.copy()

        # ------------------------------------------------------------------
        # Phase 5: Move above cylinder (planner — avoids cylinder)
        # ------------------------------------------------------------------
        print("Phase 5: Moving above cylinder...")
        cylinder_height = 0.04
        target_drop_pos = [
            tray_pos[0],
            tray_pos[1],
            tray_pos[2] + cylinder_height + CUBE_LENGTH / 2 + 0.02,
        ]
        print(f"  Target: {[f'{v:.4f}' for v in target_drop_pos]}")

        orn_drop = best_down_orn(robot, target_drop_pos)
        move_with_planner(
            planner2, robot,
            target_drop_pos, orn_drop,
            **common_kwargs,
        )
        step5_history = state_history.copy()

        # ------------------------------------------------------------------
        # Phase 6: Open gripper
        # ------------------------------------------------------------------
        print("Phase 6: Opening gripper...")
        interpolate_gripper(robot, target_angle=0.0, **common_kwargs)
        step6_history = state_history.copy()
        planner2.cleanup()

        # ------------------------------------------------------------------
        # Phase 7: Settle
        # ------------------------------------------------------------------
        print("Phase 7: Waiting for cube to settle...")
        for _ in range(500):
            p.stepSimulation()

        # ------------------------------------------------------------------
        # Success check
        # ------------------------------------------------------------------
        cube_final_pos = p.getBasePositionAndOrientation(cube_id)[0]
        dist_to_center = np.linalg.norm(
            np.array(cube_final_pos[:2]) - np.array(tray_pos[:2])
        )
        cylinder_top_z = tray_pos[2] + cylinder_height
        inside_tray = (dist_to_center < target_radius and
                       cube_final_pos[2] > cylinder_top_z + 0.01)

        if inside_tray:
            print(f"\n{'='*60}")
            print(f"🎉 SUCCESS!  Cube on cylinder at {cube_final_pos}")
            print(f"{'='*60}\n")

            final_folder = os.path.join(base_save_dir,
                                         f"iter_{successful_iterations:04d}")
            if os.path.exists(final_folder):
                shutil.rmtree(final_folder)
            os.rename(temp_folder, final_folder)

            # ── State-action arrays ────────────────────────────────────
            agent_pos = np.array(state_history)
            actions   = np.zeros_like(agent_pos)
            if len(agent_pos) > 1:
                actions[:-1] = agent_pos[1:]
                actions[-1]  = agent_pos[-1]

            # Filter static frames
            dists     = np.linalg.norm(actions - agent_pos, axis=1)
            keep_mask = dists >= 0.008
            if len(keep_mask) > 0:
                keep_mask[-1] = True

            original_count   = len(agent_pos)
            agent_pos        = agent_pos[keep_mask]
            actions          = actions[keep_mask]
            cube_positions   = np.array(cube_pos_history)
            if len(cube_positions) == original_count:
                cube_positions = cube_positions[keep_mask]

            print(f"  Filtered: {original_count} -> {len(agent_pos)} frames")

            # Prune saved files to match kept indices
            removed_indices = set(np.where(~keep_mask)[0])
            if removed_indices:
                prune_dirs = {
                    "tp_rgb":   (os.path.join(final_folder, "third_person", "rgb"),
                                 "tp_rgb_{:04d}.png"),
                    "tp_depth": (os.path.join(final_folder, "third_person", "depth"),
                                 "tp_depth_{:04d}.npy"),
                    "tp_pcd":   (os.path.join(final_folder, "third_person", "pcd"),
                                 "tp_pcd_{:04d}.npy"),
                    "tp_ply":   (os.path.join(final_folder, "third_person", "pcd"),
                                 "tp_pcd_{:04d}.ply"),
                    "tp_seg":   (os.path.join(final_folder, "third_person", "segmentation"),
                                 "tp_seg_{:04d}.npy"),
                    "cam_pose": (os.path.join(final_folder, "camera_poses"),
                                 "pose_{:04d}.json"),
                }
                for label, (dir_path, pattern) in prune_dirs.items():
                    if not os.path.isdir(dir_path):
                        continue
                    for idx in removed_indices:
                        fp = os.path.join(dir_path, pattern.format(idx))
                        if os.path.exists(fp):
                            os.remove(fp)
                    ext       = pattern.split(".")[-1]
                    surviving = sorted(
                        f for f in os.listdir(dir_path) if f.endswith(f".{ext}")
                    )
                    for new_idx, old_name in enumerate(surviving):
                        new_name = pattern.format(new_idx)
                        if old_name != new_name:
                            os.rename(
                                os.path.join(dir_path, old_name),
                                os.path.join(dir_path, new_name),
                            )
                print(f"  Pruned files: kept {len(agent_pos)}/{original_count}")

            # ── Save metadata ──────────────────────────────────────────
            state_action_data = {
                "agent_pos":        agent_pos.tolist(),
                "action":           actions.tolist(),
                "cube_pos":         cube_positions.tolist(),
                "num_frames":       len(agent_pos),
                "state_dim":        8,   # 7 arm + 1 gripper
                "cube_dim":         7,
                "success":          True,
                "attempt_number":   total_attempts,
                "success_number":   successful_iterations,
                "cube_start_pos":   cube_start_pos,
                "cube_final_pos":   list(cube_final_pos),
                "cylinder_pos":     tray_pos,
                "state_description": {
                    "arm_joints": list(range(7)),
                    "gripper":    [7],
                },
                "action_description": "Absolute target state (next state) for each timestep",
                "cube_description": {"position": [0, 1, 2], "orientation": [3, 4, 5, 6]},
            }
            with open(os.path.join(final_folder, "state_action.json"), "w") as f:
                json.dump(state_action_data, f, indent=2)

            np.save(os.path.join(final_folder, "agent_pos.npy"), agent_pos)
            np.save(os.path.join(final_folder, "actions.npy"),   actions)
            np.save(os.path.join(final_folder, "cube_pos.npy"),  cube_positions)

            np.savetxt(os.path.join(final_folder, "agent_pos.txt"), agent_pos,
                       fmt="%.6f", header="7 joint angles + gripper normalised (8 dims)")
            np.savetxt(os.path.join(final_folder, "actions.txt"), actions,
                       fmt="%.6f", header="Absolute target state t+1 (8 dims)")
            np.savetxt(os.path.join(final_folder, "cube_pos.txt"), cube_positions,
                       fmt="%.6f", header="Cube pos (xyz) + quat (xyzw) = 7 dims")

            for step_idx, hist in enumerate(
                [step1_history, step2_history, step3_history,
                 step4_history, step5_history, step6_history], start=1
            ):
                np.save(os.path.join(final_folder, f"step{step_idx}_history.npy"),
                        hist)

            print(f"Saved to: {final_folder}  "
                  f"(frames={frame_counter[0]}, states={agent_pos.shape})")
            successful_iterations += 1

        else:
            print(f"\n{'='*60}")
            print(f"❌ FAILED — Cube at {cube_final_pos}")
            print(f"{'='*60}\n")
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)
            failed_attempts += 1

        # Remove bodies for next iteration
        for bid in [cube_id, cylinder_id] + obstacle_ids:
            try:
                p.removeBody(bid)
            except Exception:
                pass
        cube_id = cylinder_id = None
        obstacle_ids = []

        total_attempts += 1

        print(f"\n{'='*60}  PROGRESS  {'='*60}")
        print(f"  Attempts: {total_attempts}  |  "
              f"Successful: {successful_iterations}/150  |  "
              f"Failed: {failed_attempts}  |  "
              f"Rate: {100*successful_iterations/total_attempts:.1f}%")
        print(f"{'='*60}\n")

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE!")
    print(f"  Saved {successful_iterations} trajectories in {total_attempts} attempts.")
    print("=" * 60 + "\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    EXCLUDE_TABLE = True

    _, table_id, plane_id = setup_simulation(freq=60, gui=True)

    robot = FrankaRobot([0, 0, 0.62], [0, 0, 0])
    robot.load()

    # Use the grasptarget link as EEF if present (better for grasp alignment)
    for i in range(p.getNumJoints(robot.id)):
        link_name = p.getJointInfo(robot.id, i)[12].decode("utf-8")
        if link_name == "panda_grasptarget":
            robot.eef_id = i
            print(f"EEF overridden to grasptarget link at index {i}")
            break

    move_and_grab_cube(robot, table_id, plane_id, EXCLUDE_TABLE=EXCLUDE_TABLE)


if __name__ == "__main__":
    main()