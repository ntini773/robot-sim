"""
Robot module - Base class for robotic manipulators.
Extends Agent with robot-specific functionality.
Replicates ArticuBot's manipulation/robot.py architecture.
"""

import numpy as np
import pybullet as p
from typing import Optional, List, Tuple, Union

from agent import Agent


class Robot(Agent):
    """
    Base class for robotic manipulators.
    Extends Agent with arm and gripper specific functionality.
    """
    
    def __init__(self, physics_client_id: int = 0):
        """
        Initialize the robot.
        
        Args:
            physics_client_id: PyBullet physics client ID
        """
        super().__init__(physics_client_id)
        
        # Robot-specific configuration (to be set by subclasses)
        self.arm_joint_indices: List[int] = []
        self.arm_num_dofs: int = 0
        self.end_effector: int = -1
        self.gripper_indices: List[int] = []
        
        # Control parameters
        self.max_velocity: float = 3.0
        self.gripper_force: float = 100.0
        
        # Rest poses
        self.arm_rest_poses: np.ndarray = np.array([])
    
    def load(self, base_position: List[float], base_orientation: Optional[List[float]] = None,
             fixed_base: bool = True) -> int:
        """
        Load the robot URDF.
        Must be implemented by subclasses.
        
        Args:
            base_position: Base position [x, y, z]
            base_orientation: Base orientation as Euler angles [roll, pitch, yaw]
            fixed_base: Whether to fix the base
            
        Returns:
            Robot body ID
        """
        raise NotImplementedError("Subclasses must implement load()")
    
    def reset_to_rest_pose(self):
        """
        Reset arm joints to rest poses.
        """
        if len(self.arm_rest_poses) > 0:
            self.set_joint_angles(self.arm_joint_indices, self.arm_rest_poses)
    
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current end-effector position and orientation.
        
        Returns:
            Tuple of (position [3], orientation quaternion [4])
        """
        return self.get_pos_orient(self.end_effector)
    
    def move_to_pose(self, target_pos: np.ndarray, target_orient: Optional[np.ndarray] = None) -> bool:
        """
        Compute IK and set arm joints to reach target pose.
        
        Args:
            target_pos: Target position [x, y, z]
            target_orient: Target orientation quaternion [x, y, z, w]
            
        Returns:
            True if IK solution found and applied
        """
        ik_solution = self.ik(target_pos, target_orient, self.end_effector)
        arm_angles = np.array(ik_solution)[:self.arm_num_dofs]
        
        # Check if valid
        if np.any(np.isnan(arm_angles)):
            return False
        
        # Apply control
        self.control(self.arm_joint_indices, arm_angles)
        return True
    
    def set_arm_joints(self, joint_angles: Union[List[float], np.ndarray]):
        """
        Set arm joint angles using position control.
        
        Args:
            joint_angles: Target angles for arm joints
        """
        forces = [self.joint_info[idx]['max_force'] for idx in self.arm_joint_indices]
        
        for i, joint_id in enumerate(self.arm_joint_indices):
            p.setJointMotorControl2(
                self.body,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=joint_angles[i],
                force=forces[i],
                maxVelocity=self.max_velocity,
                physicsClientId=self.id
            )
    
    def get_arm_joint_angles(self) -> np.ndarray:
        """
        Get current arm joint angles.
        
        Returns:
            Array of arm joint angles
        """
        return self.get_joint_angles(self.arm_joint_indices)
    
    def get_gripper_state(self) -> float:
        """
        Get current gripper state.
        Must be implemented by subclasses with gripper.
        
        Returns:
            Gripper opening value (0 = closed, 1 = open typically)
        """
        raise NotImplementedError("Subclasses must implement get_gripper_state()")
    
    def set_gripper(self, value: float):
        """
        Set gripper opening.
        Must be implemented by subclasses with gripper.
        
        Args:
            value: Target gripper value
        """
        raise NotImplementedError("Subclasses must implement set_gripper()")
    
    def get_state(self) -> np.ndarray:
        """
        Get complete robot state.
        
        Returns:
            Array containing: EE position (3), EE orientation (3 Euler), 
            arm joint angles (arm_num_dofs), gripper state (1)
        """
        ee_pos, ee_orient = self.get_end_effector_pose()
        ee_euler = np.array(p.getEulerFromQuaternion(ee_orient))
        arm_angles = self.get_arm_joint_angles()
        
        try:
            gripper = self.get_gripper_state()
        except NotImplementedError:
            gripper = 0.0
        
        return np.concatenate([ee_pos, ee_euler, arm_angles, [gripper]])
    
    def set_friction(self, friction: float = 5.0):
        """
        Set friction for gripper links.
        
        Args:
            friction: Friction coefficient
        """
        for link_idx in self.gripper_indices:
            p.changeDynamics(
                self.body, link_idx,
                lateralFriction=friction,
                rollingFriction=friction,
                spinningFriction=friction,
                physicsClientId=self.id
            )
