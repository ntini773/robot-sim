"""
Camera utilities for ArticuBot-style simulation.
Handles view matrix computation, rendering, and camera parameter extraction.
"""

import numpy as np
import pybullet as p
from typing import Tuple, List, Optional, Dict

def setup_camera(camera_eye: List[float], camera_target: List[float], 
                 camera_up: List[float] = [0, 0, 1]) -> Tuple[List[float], List[float]]:
    """
    Compute view and projection matrices from eye, target, and up vectors.
    
    Args:
        camera_eye: Camera position [x, y, z]
        camera_target: Camera look-at target [x, y, z]
        camera_up: Camera up vector [x, y, z]
        
    Returns:
        Tuple of (view_matrix, projection_matrix) could be constructed separate if needed,
        but typically we just return view matrix here or used within render.
    """
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_eye,
        cameraTargetPosition=camera_target,
        cameraUpVector=camera_up
    )
    return view_matrix

def setup_camera_rpy(camera_target: List[float], distance: float, 
                     yaw: float, pitch: float, roll: float = 0.0,
                     up_axis_index: int = 2) -> List[float]:
    """
    Compute view matrix from target, distance, and Euler angles.
    
    Args:
        camera_target: Look-at target [x, y, z]
        distance: Distance from target
        yaw: Yaw angle in degrees
        pitch: Pitch angle in degrees
        roll: Roll angle in degrees
        up_axis_index: Up axis (1=y, 2=z)
        
    Returns:
        View matrix list
    """
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        upAxisIndex=up_axis_index
    )
    return view_matrix

def get_projection_matrix(fov: float = 60.0, aspect: float = 1.0, 
                          near_val: float = 0.01, far_val: float = 100.0) -> List[float]:
    """
    Compute projection matrix.
    
    Args:
        fov: Field of view in degrees
        aspect: Aspect ratio (width/height)
        near_val: Near clipping plane
        far_val: Far clipping plane
        
    Returns:
        Projection matrix list
    """
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=near_val,
        farVal=far_val
    )
    return projection_matrix

def render(width: int, height: int, view_matrix: List[float], 
           projection_matrix: List[float], 
           physics_client_id: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render camera image.
    
    Args:
        width: Image width
        height: Image height
        view_matrix: View matrix
        projection_matrix: Projection matrix
        physics_client_id: PyBullet client ID
        
    Returns:
        Tuple of (rgb_image, depth_image, segmentation_mask)
        RGB is [H, W, 3] uint8
        Depth is [H, W] float
        Segmask is [H, W] int
    """
    _, _, rgb, depth, seg = p.getCameraImage(
        width, height,
        view_matrix,
        projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        physicsClientId=physics_client_id
    )
    
    rgb = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    depth = np.array(depth).reshape(height, width)
    seg = np.array(seg).reshape(height, width)
    
    return rgb, depth, seg

def get_wrist_camera_params(robot, offset: List[float] = [-0.05, 0, 0.12], 
                           look_at_offset: List[float] = [0.35, 0, -0.15]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get wrist camera parameters based on robot end-effector.
    
    Args:
        robot: Robot instance
        offset: Camera position offset from EE in EE frame
        look_at_offset: Look-at target offset from camera in EE frame
        
    Returns:
        Tuple of (camera_pos, camera_target, camera_up)
    """
    ee_pos, ee_orn = robot.get_end_effector_pose()
    rot_matrix = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    
    cam_pos = ee_pos + rot_matrix @ np.array(offset)
    cam_target = cam_pos + rot_matrix @ np.array(look_at_offset)
    cam_up = rot_matrix[:, 2] # Up vector is typically Z axis of EE
    
    return cam_pos, cam_target, cam_up
