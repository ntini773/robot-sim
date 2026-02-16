"""
XArm6 Pick-and-Place Gym Environment

A gym.Env wrapper for the XArm6 + Robotiq85 gripper simulation.
Actions are absolute joint angles (6D) + gripper delta (1D).

Gripper delta logic:
  - delta > 0.1  → close gripper (until normalized state >= 1.0 or raw angle > 0.35)
  - delta < -0.05 → open gripper
  - otherwise     → hold current state
"""

import os
import gym
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import random
import time
import fpsample
from collections import namedtuple
from termcolor import cprint
from gym import spaces


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
CUBE_LENGTH = 0.05
GRIPPER_CLOSE_TARGET = 0.5     # target angle when closing
GRIPPER_OPEN_TARGET = 0.0      # target angle when opening
GRIPPER_MAX_ANGLE = 0.35       # normalization cap
GRIPPER_CLOSE_DELTA_THRESH = 0.1
GRIPPER_OPEN_DELTA_THRESH = -0.05


# ─────────────────────────────────────────────
#  Utility functions
# ─────────────────────────────────────────────

def depth_to_point_cloud(depth_buffer, view_matrix, proj_matrix, base_pos, width=224, height=224):
    """
    Convert depth buffer to 3D point cloud in robot base frame.
    Uses PyBullet's view + projection matrices to unproject
    depth pixels from NDC directly to world space, then to base frame.
    """
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

    points_base = points_world - np.array(base_pos)
    return points_base


def farthest_point_sampling(points, n_samples, colors=None):
    """
    Farthest Point Sampling using fpsample (Rust-backed).
    """
    if len(points) <= n_samples:
        return (points, colors) if colors is not None else (points, None)

    indices = fpsample.fps_npdu_sampling(points, n_samples)
    sampled_points = points[indices]
    sampled_colors = colors[indices] if colors is not None else None
    return sampled_points, sampled_colors


def create_cylinder(radius, height, pos, color=[0.7, 0.7, 0.7, 1]):
    """Creates a static cylinder body using primitives."""
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color
    )
    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius, height=height
    )
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[pos[0], pos[1], pos[2] + height / 2],
    )
    return body_id


# ─────────────────────────────────────────────
#  Robot class
# ─────────────────────────────────────────────

class XArm6Robotiq85:
    """XArm6 Robot with Robotiq 85 Gripper"""

    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 6
        self.arm_num_dofs = 6
        self.arm_rest_poses = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
        self.gripper_range = [0, 0.085]
        self.max_velocity = 3

    def load(self):
        self.id = p.loadURDF(
            "./urdf/xarm6_robotiq_85.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
        )
        self._parse_joint_info()
        self._setup_mimic_joints()

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
            jid = info[0]
            jname = info[1].decode("utf-8")
            jtype = info[2]
            controllable = jtype != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jid)
            self.joints.append(
                JointInfo(jid, jname, jtype, info[8], info[9], info[10], info[11], controllable)
            )

        self.arm_controllable_joints = self.controllable_joints[: self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][: self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][: self.arm_num_dofs]
        self.arm_joint_ranges = [
            ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)
        ]

    def _setup_mimic_joints(self):
        """Setup mimic joints for Robotiq gripper with strong constraints."""
        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }

        self.mimic_parent_id = [
            j.id for j in self.joints if j.name == mimic_parent_name
        ][0]

        self.mimic_child_multiplier = {
            j.id: mimic_children_names[j.name]
            for j in self.joints
            if j.name in mimic_children_names
        }

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(
                self.id, self.mimic_parent_id,
                self.id, joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100000, erp=1)

        # Increase friction for gripper pads
        gripper_links = [self.mimic_parent_id] + list(self.mimic_child_multiplier.keys())
        for link_id in gripper_links:
            p.changeDynamics(
                self.id, link_id,
                lateralFriction=5.0, spinningFriction=5.0, rollingFriction=5.0,
            )

    # ── State queries ──

    def get_gripper_normalized(self):
        """Return gripper state normalized to [0, 1] (capped at 0.35 rad)."""
        raw = p.getJointState(self.id, self.mimic_parent_id)[0]
        if abs(raw) < 1e-3:
            raw = 0.0
        return min(raw, GRIPPER_MAX_ANGLE) / GRIPPER_MAX_ANGLE

    def get_gripper_raw_angle(self):
        """Return raw gripper finger_joint angle."""
        return p.getJointState(self.id, self.mimic_parent_id)[0]

    def get_joint_positions(self):
        """Return 7D array: [arm_joints(6), gripper_normalized(1)]."""
        arm = [p.getJointState(self.id, jid)[0] for jid in self.arm_controllable_joints]
        return np.array(arm + [self.get_gripper_normalized()])

    def get_robot_state(self):
        """Alias for get_joint_positions (7D)."""
        return self.get_joint_positions()

    def get_eef_position(self):
        """Get end-effector 3D position."""
        return np.array(p.getLinkState(self.id, self.eef_id)[0])

    # ── Control ──

    def set_arm_joints(self, joint_positions):
        """Set arm joint positions via position control."""
        for i, jid in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id, jid, p.POSITION_CONTROL,
                joint_positions[i], maxVelocity=self.max_velocity,
            )

    def command_gripper_close(self, target_angle=GRIPPER_CLOSE_TARGET, force=500):
        """Send a single close command to gripper (position control)."""
        p.setJointMotorControl2(
            self.id, self.mimic_parent_id, p.POSITION_CONTROL,
            targetPosition=target_angle, force=force,
        )
        for jid, mult in self.mimic_child_multiplier.items():
            p.setJointMotorControl2(
                self.id, jid, p.POSITION_CONTROL,
                targetPosition=target_angle * mult, force=force,
            )

    def command_gripper_open(self, target_angle=GRIPPER_OPEN_TARGET, force=500):
        """Send a single open command to gripper (position control)."""
        p.setJointMotorControl2(
            self.id, self.mimic_parent_id, p.POSITION_CONTROL,
            targetPosition=target_angle, force=force,
        )
        for jid, mult in self.mimic_child_multiplier.items():
            p.setJointMotorControl2(
                self.id, jid, p.POSITION_CONTROL,
                targetPosition=target_angle * mult, force=force,
            )

    def reset_posture(self):
        """Teleport robot to deterministic rest pose with gripper open."""
        initial_pos_world = [0.125, 0.01, 0.62 + 0.377]
        initial_orn_world = p.getQuaternionFromEuler([3.14, 0, 0])

        # Reset joints to rest poses for deterministic IK seed
        for i, jid in enumerate(self.arm_controllable_joints):
            p.resetJointState(self.id, jid, self.arm_rest_poses[i], targetVelocity=0)

        target_joints = p.calculateInverseKinematics(
            self.id, self.eef_id, initial_pos_world, initial_orn_world,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
        )

        # Disable motors to avoid fighting during teleport
        for j in range(p.getNumJoints(self.id)):
            p.setJointMotorControl2(self.id, j, p.VELOCITY_CONTROL, force=0)

        for i, jid in enumerate(self.arm_controllable_joints):
            p.resetJointState(self.id, jid, target_joints[i], targetVelocity=0)
            p.setJointMotorControl2(
                self.id, jid, p.POSITION_CONTROL,
                targetPosition=target_joints[i], force=1000,
            )

        # Open gripper
        p.resetJointState(self.id, self.mimic_parent_id, 0.0, targetVelocity=0)
        p.setJointMotorControl2(
            self.id, self.mimic_parent_id, p.POSITION_CONTROL,
            targetPosition=0.0, force=500,
        )
        for jid, mult in self.mimic_child_multiplier.items():
            p.resetJointState(self.id, jid, 0.0, targetVelocity=0)
            p.setJointMotorControl2(
                self.id, jid, p.POSITION_CONTROL,
                targetPosition=0.0, force=500,
            )

        for _ in range(100):
            p.stepSimulation()


# ─────────────────────────────────────────────
#  Gym Environment
# ─────────────────────────────────────────────

class XArm6PickPlaceEnv(gym.Env):
    """
    PyBullet XArm6 Pick-and-Place Environment.

    Action: 7D — [target_joint_angles(6), gripper_delta(1)]
      - Joints 0-5: absolute target positions sent via position control
      - Joint 6: gripper delta → triggers close/open/hold logic

    Observation: Dict with point_cloud, agent_pos, image
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        use_gui=False,
        num_points=2500,
        image_size=224,
        action_dim=7,
        sim_freq=60,
        sim_substeps=1,
        max_steps=350,
        capture_table=False,
    ):
        self.use_gui = use_gui
        self.num_points = num_points
        self.image_size = image_size
        self.max_steps = max_steps
        self.current_step = 0
        self.action_dim = action_dim
        self.capture_table = capture_table
        self.sim_freq = sim_freq
        self.sim_substeps = sim_substeps

        cprint(f"XArm6PickPlaceEnv initialized — action_dim={action_dim}, "
               f"num_points={num_points}, sim_freq={sim_freq}", "cyan")

        # ── Connect to PyBullet ──
        mode = p.GUI if self.use_gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1 / self.sim_freq)

        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useMaximalCoordinates=True)
        self.table_id = p.loadURDF(
            "table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0])
        )

        # ── Load robot ──
        self.robot = XArm6Robotiq85([0, 0, 0.62], [0, 0, 0])
        self.robot.load()

        # ── Camera (same as data collection) ──
        self.tp_cam_eye = [0.7463, 0.3093, 0.62 + 0.5574]      # [0.7463, 0.3093, 1.1774]
        self.tp_cam_target = [0.7463 - 0.7751, 0.3093 - 0.4045, 1.1774 - 0.4855]
        self.tp_cam_up = [0, 0, 1]

        # ── Objects ──
        self.cube_id = None
        self.cylinder_id = None
        self.cube_start_pos = None
        self.cylinder_pos = None

        # ── Spaces ──
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "point_cloud": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_points, 3), dtype=np.float32,
            ),
            "agent_pos": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.action_dim,), dtype=np.float32,
            ),
            "image": spaces.Box(
                low=0, high=255,
                shape=(3, self.image_size, self.image_size), dtype=np.uint8,
            ),
        })

        self.is_success_flag = False

    # ──────────── reset ────────────

    def reset(self, cube_start_pos=None, cube_start_orn=None,
              cylinder_pos=None, cylinder_color=None):
        """
        Reset the environment.

        Args:
            cube_start_pos: [x, y, z] or None (random)
            cube_start_orn: quaternion [x,y,z,w] or euler [r,p,y] or None
            cylinder_pos: [x, y, z] or None (random)
            cylinder_color: [r, g, b, a] or None (default orange)
        """
        self.current_step = 0
        self.is_success_flag = False

        # Clean up previous objects
        if self.cube_id is not None:
            p.removeBody(self.cube_id)
            self.cube_id = None
        if self.cylinder_id is not None:
            p.removeBody(self.cylinder_id)
            self.cylinder_id = None

        # Reset robot
        self.robot.reset_posture()

        # Settle
        for _ in range(100):
            p.stepSimulation()

        # ── Spawn cylinder ──
        if cylinder_pos is not None:
            self.cylinder_pos = list(cylinder_pos)
        else:
            self.cylinder_pos = [
                random.uniform(0.20, 0.35),
                random.uniform(0.15, 0.25),
                0.625,
            ]

        cyl_radius = 0.05
        cyl_height = 0.04
        cyl_color = cylinder_color if cylinder_color is not None else [1, 0.5, 0, 1]
        self.cylinder_id = create_cylinder(
            cyl_radius, cyl_height, self.cylinder_pos, color=cyl_color
        )

        # ── Spawn cube ──
        if cube_start_pos is not None:
            self.cube_start_pos = list(cube_start_pos)
        else:
            self.cube_start_pos = [
                random.uniform(0.15, 0.35),
                random.uniform(-0.2, 0.1),
                0.65,
            ]

        if cube_start_orn is not None:
            if len(cube_start_orn) == 4:
                cube_orn_quat = cube_start_orn
            else:
                cube_orn_quat = p.getQuaternionFromEuler(cube_start_orn)
        else:
            cube_orn_quat = p.getQuaternionFromEuler([0, 0, 0])

        self.cube_id = p.loadURDF("cube_small.urdf", self.cube_start_pos, cube_orn_quat)
        p.changeDynamics(self.cube_id, -1,
                         lateralFriction=5.0, spinningFriction=5.0, rollingFriction=5.0)
        p.changeVisualShape(self.cube_id, -1, rgbaColor=[0, 0, 0, 1])

        # Settle after spawning
        for _ in range(50):
            p.stepSimulation()

        return self._get_obs()

    # ──────────── step ────────────

    def step(self, action):
        """
        Execute one environment step.

        Args:
            action: 7D array [target_joints(6), gripper_delta(1)]

        Returns:
            obs, reward, done, info
        """
        self.current_step += 1
        action = np.asarray(action, dtype=np.float32)

        # 1) Set arm joints (absolute targets)
        target_joints = action[:6]
        self.robot.set_arm_joints(target_joints)

        # 2) Gripper delta logic
        gripper_delta = action[6] if len(action) > 6 else 0.0
        self._apply_gripper_delta(gripper_delta)

        # 3) Step simulation
        for _ in range(self.sim_substeps):
            p.stepSimulation()

        # 4) Observation
        obs = self._get_obs()

        # 5) Reward / done
        reward = 0.0
        done = False

        if self.cube_id is not None:
            cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
            cyl_height = 0.04
            cyl_radius = 0.05
            cyl_top_z = self.cylinder_pos[2] + cyl_height
            dist_xy = np.linalg.norm(
                np.array(cube_pos[:2]) - np.array(self.cylinder_pos[:2])
            )
            if dist_xy < cyl_radius and cube_pos[2] > cyl_top_z + 0.01:
                reward = 1.0
                self.is_success_flag = True
                done = True

        if self.current_step >= self.max_steps:
            done = True

        info = {"is_success": self.is_success_flag}
        return obs, reward, done, info

    # ──────────── gripper logic ────────────

    def _apply_gripper_delta(self, gripper_delta):
        """
        Apply gripper delta logic:
          delta > 0.1   → close (iterate until normalized >= 1.0 or raw > 0.35)
          delta < -0.05 → open
          else          → hold
        """
        if gripper_delta > GRIPPER_CLOSE_DELTA_THRESH:
            # Close: keep sending close commands until fully closed
            max_iters = 100
            for _ in range(max_iters):
                self.robot.command_gripper_close()
                p.stepSimulation()
                raw_angle = self.robot.get_gripper_raw_angle()
                norm_state = self.robot.get_gripper_normalized()
                if norm_state >= 1.0 or raw_angle > GRIPPER_MAX_ANGLE:
                    break

        elif gripper_delta < GRIPPER_OPEN_DELTA_THRESH:
            # Open: send open command and let simulation settle
            self.robot.command_gripper_open()
            max_iters = 100
            for _ in range(max_iters):
                p.stepSimulation()
                raw_angle = self.robot.get_gripper_raw_angle()
                if raw_angle <= 0.01:
                    break
        # else: hold — do nothing

    # ──────────── observation ────────────

    def _get_obs(self):
        """Build observation dict: point_cloud, agent_pos, image."""
        agent_pos = self.robot.get_joint_positions()  # 7D

        # ── Camera ──
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.tp_cam_eye,
            cameraTargetPosition=self.tp_cam_target,
            cameraUpVector=self.tp_cam_up,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.01, farVal=3.0
        )

        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            self.image_size, self.image_size,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )

        rgb_img = np.array(rgb_img)[:, :, :3]
        depth_buffer = np.array(depth_img)
        seg_img = np.array(seg_img)

        # ── Exclusion masks ──
        exclude_ids = [self.plane_id]
        if not self.capture_table and self.table_id is not None:
            exclude_ids.append(self.table_id)

        exclude_mask = np.zeros_like(seg_img, dtype=bool)
        for oid in exclude_ids:
            exclude_mask |= (seg_img == oid)

        background_mask = (depth_buffer >= 0.9999).flatten()

        # ── Point cloud ──
        pc = depth_to_point_cloud(
            depth_buffer, view_matrix, proj_matrix,
            self.robot.base_pos, self.image_size, self.image_size,
        )
        pc_flat = pc.reshape(-1, 3)

        valid = (pc_flat[:, 2] < 2.5) & (~exclude_mask.flatten()) & (~background_mask)
        pc_filtered = pc_flat[valid]

        # FPS downsample
        if pc_filtered.shape[0] == 0:
            pc_filtered = np.zeros((self.num_points, 3), dtype=np.float32)
        elif pc_filtered.shape[0] > self.num_points:
            pc_filtered, _ = farthest_point_sampling(pc_filtered, self.num_points)
        elif pc_filtered.shape[0] < self.num_points:
            pad = np.zeros((self.num_points - pc_filtered.shape[0], 3), dtype=np.float32)
            pc_filtered = np.vstack([pc_filtered, pad])

        # ── Image (C, H, W) ──
        rgb_chw = rgb_img.transpose(2, 0, 1)

        return {
            "point_cloud": pc_filtered.astype(np.float32),
            "agent_pos": agent_pos.astype(np.float32),
            "image": rgb_chw.astype(np.uint8),
        }

    # ──────────── render / close / utils ────────────

    def render(self, mode="rgb_array"):
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.tp_cam_eye,
            cameraTargetPosition=self.tp_cam_target,
            cameraUpVector=self.tp_cam_up,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.01, farVal=3.0
        )
        _, _, rgb_img, _, _ = p.getCameraImage(
            self.image_size, self.image_size,
            viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        )
        return np.array(rgb_img)[:, :, :3].astype(np.uint8)

    def close(self):
        p.disconnect(self.physics_client)

    def is_success(self):
        return self.is_success_flag

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
