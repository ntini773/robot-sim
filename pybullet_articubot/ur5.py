"""
UR5 Robotiq 85 module - Specific implementation for UR5 with Robotiq 85 gripper.
Replicates ArticuBot's manipulation/panda.py architecture but for UR5.
"""

import os
import numpy as np
import pybullet as p
from typing import Optional, List, Tuple

from robot import Robot


class UR5Robotiq85(Robot):
    """
    UR5 robot with Robotiq 85 parallel jaw gripper.
    """
    
    def __init__(self, physics_client_id: int = 0):
        """
        Initialize UR5 Robotiq 85.
        
        Args:
            physics_client_id: PyBullet physics client ID
        """
        super().__init__(physics_client_id)
        
        # UR5 arm configuration (6 DOF)
        self.arm_num_dofs = 6
        self.arm_rest_poses = np.array([-1.57, -1.54, 1.34, -1.37, -1.57, 0.0])
        
        # Control parameters
        self.max_velocity = 3.0
        self.gripper_force = 1500.0
        self.gripper_max_velocity = 1.5
        
        # Gripper configuration
        self.gripper_range = [0.0, 0.085]  # Robotiq 85 range
        self.gripper_open_angle = 0.0
        self.gripper_closed_angle = 0.8
        
        # Mimic joint configuration for Robotiq gripper
        self.mimic_parent_id: Optional[int] = None
        self.mimic_child_multiplier: dict = {}
    
    def load(self, base_position: List[float], 
             base_orientation: Optional[List[float]] = None,
             fixed_base: bool = True) -> int:
        """
        Load UR5 Robotiq 85 URDF.
        
        Args:
            base_position: Base position [x, y, z]
            base_orientation: Base orientation as Euler angles [roll, pitch, yaw]
            fixed_base: Whether to fix the base
            
        Returns:
            Robot body ID
        """
        if base_orientation is None:
            base_orientation = [0, 0, 0]
        
        base_orient_quat = p.getQuaternionFromEuler(base_orientation)
        
        # Get URDF path (relative to this file or absolute)
        urdf_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(urdf_dir, "assets", "ur5_robotiq_85", "ur5_robotiq_85.urdf")
        
        # Fallback to the existing URDF location
        if not os.path.exists(urdf_path):
            # Try relative to workspace
            urdf_path = os.path.join(
                os.path.dirname(urdf_dir), 
                "pybullet", "urdf", "ur5_robotiq_85.urdf"
            )
        
        self.body = p.loadURDF(
            urdf_path,
            basePosition=base_position,
            baseOrientation=base_orient_quat,
            useFixedBase=fixed_base,
            physicsClientId=self.id
        )
        
        # Parse joint info
        self._parse_joint_info()
        
        # Configure UR5-specific joints
        self._configure_joints()
        
        # Setup Robotiq mimic joints
        self._setup_mimic_joints()
        
        # Update IK limits
        self.update_ik_limits(self.arm_joint_indices)
        
        # Set rest poses for IK
        self.ik_rest_poses = self.arm_rest_poses.copy()
        
        return self.body
    
    def _configure_joints(self):
        """
        Configure UR5-specific joint indices based on the URDF.
        """
        # Find arm joints (first 6 controllable joints are typically the arm)
        self.arm_joint_indices = self.controllable_joints[:self.arm_num_dofs]
        
        # End-effector is typically link 7 for UR5
        self.end_effector = 7
        
        # Find gripper joints
        for joint_id, info in self.joint_info.items():
            name = info['name'].lower()
            if 'finger' in name or 'gripper' in name:
                if 'finger_joint' == info['name']:
                    self.mimic_parent_id = joint_id
                if joint_id not in self.arm_joint_indices:
                    self.gripper_indices.append(joint_id)
        
        # Get joint limits for arm
        self.arm_lower_limits = [self.joint_info[idx]['lower_limit'] 
                                  for idx in self.arm_joint_indices]
        self.arm_upper_limits = [self.joint_info[idx]['upper_limit'] 
                                  for idx in self.arm_joint_indices]
    
    def _setup_mimic_joints(self):
        """
        Setup mimic joint constraints for Robotiq 85 gripper.
        The Robotiq gripper has several joints that should move together.
        """
        # Define mimic relationships for Robotiq 85
        mimic_parent_name = "finger_joint"
        mimic_children = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }
        
        # Find parent joint ID
        for joint_id, info in self.joint_info.items():
            if info['name'] == mimic_parent_name:
                self.mimic_parent_id = joint_id
                break
        
        if self.mimic_parent_id is None:
            print("Warning: Could not find mimic parent joint 'finger_joint'")
            return
        
        # Setup mimic children
        for joint_id, info in self.joint_info.items():
            if info['name'] in mimic_children:
                self.mimic_child_multiplier[joint_id] = mimic_children[info['name']]
        
        # Create gear constraints for synchronized movement
        for child_id, multiplier in self.mimic_child_multiplier.items():
            constraint = p.createConstraint(
                self.body,
                self.mimic_parent_id,
                self.body,
                child_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
                physicsClientId=self.id
            )
            # High force and erp for stiff constraints
            p.changeConstraint(constraint, gearRatio=-multiplier, maxForce=100000, erp=1)
    
    def get_gripper_state(self) -> float:
        """
        Get normalized gripper state in [0, 1] range.
        0 = open, 1 = closed
        
        Returns:
            Normalized gripper value
        """
        if self.mimic_parent_id is None:
            return 0.0
        
        raw_angle = p.getJointState(self.body, self.mimic_parent_id, physicsClientId=self.id)[0]
        
        # Apply noise threshold
        if abs(raw_angle) < 1e-3:
            raw_angle = 0.0
        
        # Normalize to [0, 1] based on gripper range
        # 0 = open (angle = 0), 1 = closed (angle = 0.8)
        normalized = min(raw_angle, self.gripper_closed_angle) / self.gripper_closed_angle
        
        return normalized
    
    def set_gripper(self, value: float, use_position_control: bool = False):
        """
        Set gripper opening.
        
        Args:
            value: Target value (0 = open, 1 = closed) or raw angle
            use_position_control: If True, use motor control; if False, reset joint state
        """
        # Convert normalized value to angle
        if value <= 1.0:
            target_angle = value * self.gripper_closed_angle
        else:
            target_angle = value
        
        if use_position_control:
            # Use position control for smooth movement
            p.setJointMotorControl2(
                self.body,
                self.mimic_parent_id,
                p.POSITION_CONTROL,
                targetPosition=target_angle,
                force=self.gripper_force,
                maxVelocity=self.gripper_max_velocity,
                physicsClientId=self.id
            )
            
            # Also control mimic children
            for child_id, multiplier in self.mimic_child_multiplier.items():
                child_target = target_angle * multiplier
                p.setJointMotorControl2(
                    self.body,
                    child_id,
                    p.POSITION_CONTROL,
                    targetPosition=child_target,
                    force=self.gripper_force,
                    maxVelocity=self.gripper_max_velocity,
                    physicsClientId=self.id
                )
        else:
            # Instant reset (useful for initialization)
            p.resetJointState(self.body, self.mimic_parent_id, target_angle, physicsClientId=self.id)
            
            for child_id, multiplier in self.mimic_child_multiplier.items():
                p.resetJointState(self.body, child_id, target_angle * multiplier, physicsClientId=self.id)
            
            # Step simulation to apply
            for _ in range(10):
                p.stepSimulation(physicsClientId=self.id)
    
    def open_gripper(self):
        """Open the gripper fully."""
        self.set_gripper(0.0)
    
    def close_gripper(self):
        """Close the gripper fully."""
        self.set_gripper(1.0)
    
    def get_gripper_finger_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get positions of left and right gripper fingers.
        
        Returns:
            Tuple of (left_finger_pos, right_finger_pos)
        """
        left_finger_pos = None
        right_finger_pos = None
        
        for joint_id, info in self.joint_info.items():
            link_name = info.get('link_name', '').lower()
            if 'left' in link_name and 'finger' in link_name:
                pos, _ = self.get_pos_orient(joint_id)
                if 'inner' in link_name:
                    left_finger_pos = pos
            elif 'right' in link_name and 'finger' in link_name:
                pos, _ = self.get_pos_orient(joint_id)
                if 'inner' in link_name:
                    right_finger_pos = pos
        
        # Fallback to gripper indices
        if left_finger_pos is None and len(self.gripper_indices) >= 2:
            left_finger_pos, _ = self.get_pos_orient(self.gripper_indices[0])
        if right_finger_pos is None and len(self.gripper_indices) >= 2:
            right_finger_pos, _ = self.get_pos_orient(self.gripper_indices[1])
        
        if left_finger_pos is None:
            left_finger_pos = np.zeros(3)
        if right_finger_pos is None:
            right_finger_pos = np.zeros(3)
        
        return left_finger_pos, right_finger_pos
    
    def get_gripper_keypoints(self) -> np.ndarray:
        """
        Get 4 keypoints representing gripper state.
        Similar to ArticuBot's gripper point cloud representation.
        
        Returns:
            Array of shape (4, 3) with gripper keypoints
        """
        ee_pos, _ = self.get_end_effector_pose()
        left_finger, right_finger = self.get_gripper_finger_positions()
        
        # Get hand/wrist position (link before end-effector)
        hand_pos = ee_pos  # Default to EE if hand link not found
        if self.end_effector > 0:
            hand_pos, _ = self.get_pos_orient(self.end_effector - 1)
        
        return np.array([hand_pos, left_finger, right_finger, ee_pos])
