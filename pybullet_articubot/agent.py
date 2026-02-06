"""
Agent module - Base class for controllable bodies in PyBullet.
Replicates ArticuBot's manipulation/agent.py architecture.
"""

import numpy as np
import pybullet as p
from typing import Optional, List, Tuple, Union


class Agent:
    """
    Base class for any controllable agent in the simulation.
    Provides core functionality for joint control, IK solving, and state retrieval.
    """
    
    def __init__(self, physics_client_id: int = 0):
        """
        Initialize the agent.
        
        Args:
            physics_client_id: PyBullet physics client ID
        """
        self.id = physics_client_id  # Physics client ID
        self.body: Optional[int] = None  # Robot body ID (set after loading)
        
        # Joint information (populated after loading)
        self.num_joints: int = 0
        self.joint_info: dict = {}
        self.controllable_joints: List[int] = []
        self.joint_lower_limits: np.ndarray = np.array([])
        self.joint_upper_limits: np.ndarray = np.array([])
        self.joint_ranges: np.ndarray = np.array([])
        self.joint_rest_poses: np.ndarray = np.array([])
        self.joint_max_forces: np.ndarray = np.array([])
        self.joint_max_velocities: np.ndarray = np.array([])
        
        # IK configuration
        self.ik_lower_limits: np.ndarray = np.array([])
        self.ik_upper_limits: np.ndarray = np.array([])
        self.ik_joint_ranges: np.ndarray = np.array([])
        self.ik_rest_poses: np.ndarray = np.array([])
    
    def _parse_joint_info(self):
        """
        Parse joint information from the loaded URDF.
        Populates joint limits, controllable joints, etc.
        """
        self.num_joints = p.getNumJoints(self.body, physicsClientId=self.id)
        self.joint_info = {}
        self.controllable_joints = []
        
        lower_limits = []
        upper_limits = []
        max_forces = []
        max_velocities = []
        
        for i in range(self.num_joints):
            info = p.getJointInfo(self.body, i, physicsClientId=self.id)
            joint_id = info[0]
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            lower_limit = info[8]
            upper_limit = info[9]
            max_force = info[10]
            max_velocity = info[11]
            link_name = info[12].decode('utf-8')
            
            self.joint_info[joint_id] = {
                'name': joint_name,
                'type': joint_type,
                'lower_limit': lower_limit,
                'upper_limit': upper_limit,
                'max_force': max_force,
                'max_velocity': max_velocity,
                'link_name': link_name,
            }
            
            # Track controllable (non-fixed) joints
            if joint_type != p.JOINT_FIXED:
                self.controllable_joints.append(joint_id)
                lower_limits.append(lower_limit)
                upper_limits.append(upper_limit)
                max_forces.append(max_force)
                max_velocities.append(max_velocity)
        
        self.joint_lower_limits = np.array(lower_limits)
        self.joint_upper_limits = np.array(upper_limits)
        self.joint_ranges = self.joint_upper_limits - self.joint_lower_limits
        self.joint_max_forces = np.array(max_forces)
        self.joint_max_velocities = np.array(max_velocities)
    
    def update_ik_limits(self, joint_indices: List[int]):
        """
        Update IK solver limits for specified joints.
        
        Args:
            joint_indices: List of joint indices to use for IK
        """
        lower = []
        upper = []
        ranges = []
        rest = []
        
        for idx in joint_indices:
            info = self.joint_info[idx]
            lower.append(info['lower_limit'])
            upper.append(info['upper_limit'])
            ranges.append(info['upper_limit'] - info['lower_limit'])
            # Use middle of range as rest pose
            rest.append((info['lower_limit'] + info['upper_limit']) / 2)
        
        self.ik_lower_limits = np.array(lower)
        self.ik_upper_limits = np.array(upper)
        self.ik_joint_ranges = np.array(ranges)
        self.ik_rest_poses = np.array(rest)
    
    def set_joint_angles(self, joint_indices: List[int], angles: Union[List[float], np.ndarray]):
        """
        Instantly reset joints to specified angles (no dynamics).
        
        Args:
            joint_indices: List of joint indices to set
            angles: Target angles for each joint
        """
        for idx, angle in zip(joint_indices, angles):
            p.resetJointState(self.body, idx, angle, physicsClientId=self.id)
    
    def get_joint_angles(self, joint_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Get current joint angles.
        
        Args:
            joint_indices: List of joint indices (default: all controllable joints)
            
        Returns:
            Array of joint angles
        """
        if joint_indices is None:
            joint_indices = self.controllable_joints
        
        angles = []
        for idx in joint_indices:
            state = p.getJointState(self.body, idx, physicsClientId=self.id)
            angles.append(state[0])
        
        return np.array(angles)
    
    def control(self, joint_indices: List[int], target_angles: Union[List[float], np.ndarray],
                forces: Optional[Union[List[float], np.ndarray]] = None,
                max_velocities: Optional[Union[List[float], np.ndarray]] = None):
        """
        Apply position control to specified joints.
        
        Args:
            joint_indices: List of joint indices to control
            target_angles: Target angles for each joint
            forces: Optional max forces (defaults to joint max forces)
            max_velocities: Optional max velocities (defaults to joint max velocities)
        """
        if forces is None:
            forces = [self.joint_info[idx]['max_force'] for idx in joint_indices]
        if max_velocities is None:
            max_velocities = [self.joint_info[idx]['max_velocity'] for idx in joint_indices]
        
        p.setJointMotorControlArray(
            self.body,
            joint_indices,
            p.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=forces,
            positionGains=[0.03] * len(joint_indices),
            physicsClientId=self.id
        )
    
    def get_pos_orient(self, link_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get position and orientation of a link.
        
        Args:
            link_index: Link index (-1 for base)
            
        Returns:
            Tuple of (position [3], orientation quaternion [4])
        """
        if link_index == -1:
            pos, orient = p.getBasePositionAndOrientation(self.body, physicsClientId=self.id)
        else:
            state = p.getLinkState(self.body, link_index, physicsClientId=self.id)
            pos, orient = state[0], state[1]
        
        return np.array(pos), np.array(orient)
    
    def ik(self, target_pos: np.ndarray, target_orient: Optional[np.ndarray] = None,
           target_link: int = -1, max_iters: int = 100, 
           residual_threshold: float = 1e-4) -> np.ndarray:
        """
        Compute inverse kinematics using PyBullet's solver.
        
        Args:
            target_pos: Target position [x, y, z]
            target_orient: Target orientation quaternion [x, y, z, w] (optional)
            target_link: End-effector link index
            max_iters: Maximum IK iterations
            residual_threshold: Convergence threshold
            
        Returns:
            Joint angles array
        """
        target_pos = np.array(target_pos)
        if target_orient is not None:
            target_orient = np.array(target_orient)
            joint_angles = p.calculateInverseKinematics(
                self.body,
                target_link,
                targetPosition=target_pos.tolist(),
                targetOrientation=target_orient.tolist(),
                lowerLimits=self.ik_lower_limits.tolist(),
                upperLimits=self.ik_upper_limits.tolist(),
                jointRanges=self.ik_joint_ranges.tolist(),
                restPoses=self.ik_rest_poses.tolist(),
                maxNumIterations=max_iters,
                residualThreshold=residual_threshold,
                physicsClientId=self.id
            )
        else:
            joint_angles = p.calculateInverseKinematics(
                self.body,
                target_link,
                targetPosition=target_pos.tolist(),
                lowerLimits=self.ik_lower_limits.tolist(),
                upperLimits=self.ik_upper_limits.tolist(),
                jointRanges=self.ik_joint_ranges.tolist(),
                restPoses=self.ik_rest_poses.tolist(),
                maxNumIterations=max_iters,
                residualThreshold=residual_threshold,
                physicsClientId=self.id
            )
        
        return np.array(joint_angles)
    
    def ik_with_random_restart(self, target_pos: np.ndarray, 
                                target_orient: Optional[np.ndarray] = None,
                                target_link: int = -1,
                                arm_joint_indices: Optional[List[int]] = None,
                                num_attempts: int = 25,
                                position_threshold: float = 0.005) -> List[np.ndarray]:
        """
        Compute IK with multiple random restarts to find diverse solutions.
        Similar to ArticuBot's approach of sampling multiple IK solutions.
        
        Args:
            target_pos: Target position [x, y, z]
            target_orient: Target orientation quaternion (optional)
            target_link: End-effector link index
            arm_joint_indices: Joint indices for the arm
            num_attempts: Number of random restart attempts
            position_threshold: Maximum position error for valid solution
            
        Returns:
            List of valid joint angle solutions
        """
        target_pos = np.array(target_pos)
        if target_orient is not None:
            target_orient = np.array(target_orient)
            
        if arm_joint_indices is None:
            arm_joint_indices = self.controllable_joints
        
        solutions = []
        original_angles = self.get_joint_angles(arm_joint_indices)
        
        for _ in range(num_attempts):
            # Sample random starting configuration
            random_start = np.random.uniform(
                self.ik_lower_limits[:len(arm_joint_indices)],
                self.ik_upper_limits[:len(arm_joint_indices)]
            )
            
            # Set robot to random config
            self.set_joint_angles(arm_joint_indices, random_start)
            
            # Solve IK
            ik_solution = self.ik(target_pos, target_orient, target_link)
            arm_solution = np.array(ik_solution)[:len(arm_joint_indices)]
            
            # Set robot to solution and check error
            self.set_joint_angles(arm_joint_indices, arm_solution)
            actual_pos, _ = self.get_pos_orient(target_link)
            error = np.linalg.norm(actual_pos - target_pos)
            
            # Check if solution is valid
            if error < position_threshold:
                # Also check joint limits
                if np.all(arm_solution >= self.ik_lower_limits[:len(arm_joint_indices)]) and \
                   np.all(arm_solution <= self.ik_upper_limits[:len(arm_joint_indices)]):
                    solutions.append(arm_solution)
        
        # Restore original configuration
        self.set_joint_angles(arm_joint_indices, original_angles)
        
        return solutions
    
    def apply_gravity_compensation(self):
        """
        Apply gravity compensation to maintain current pose.
        Useful when simulation needs to run without movement.
        """
        for joint_id in self.controllable_joints:
            current_angle = p.getJointState(self.body, joint_id, physicsClientId=self.id)[0]
            p.setJointMotorControl2(
                self.body,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=current_angle,
                force=self.joint_info[joint_id]['max_force'],
                physicsClientId=self.id
            )
