"""
Observation wrapper.
Generates point cloud observations from simulation state.
Replicates ArticuBot's robogen_wrapper.py functionality.
"""

import gym
import numpy as np
from gym import spaces
from typing import Dict, Any, Tuple

from sim import SimpleEnv
from utils.point_cloud import depth_to_point_cloud, fps_downsample, get_object_point_cloud
from utils.camera import get_wrist_camera_params

class PointCloudWrapper(gym.ObservationWrapper):
    """
    Wrapper to provide point cloud observations.
    """
    
    def __init__(self, env: SimpleEnv, img_width=320, img_height=240, num_points=1000):
        super().__init__(env)
        self.env = env
        self.img_width = img_width
        self.img_height = img_height
        self.num_points = num_points
        
        # Extended observation space
        self.observation_space = spaces.Dict({
            'point_cloud': spaces.Box(low=-np.inf, high=np.inf, shape=(num_points, 3), dtype=np.float32),
            'agent_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
            'gripper_pcd': spaces.Box(low=-np.inf, high=np.inf, shape=(4, 3), dtype=np.float32)
        })
        
    def observation(self, obs: np.ndarray) -> Dict[str, Any]:
        """
        Compute point cloud observation from current state.
        
        Args:
            obs: Raw robot state from env
            
        Returns:
            Dictionary with point_cloud, agent_pos, gripper_pcd
        """
        # 1. Capture images from multiple views
        # View 1: Front-ish
        self.env.setup_camera(camera_target=[0, 0, 0.2], distance=1.2, yaw=45, pitch=-35)
        _, depth1, seg1 = self.env.render()
        view_matrix1 = self.env.view_matrix
        proj_matrix1 = self.env.projection_matrix
        
        # View 2: Side-ish
        self.env.setup_camera(camera_target=[0, 0, 0.2], distance=1.2, yaw=135, pitch=-35)
        _, depth2, seg2 = self.env.render()
        view_matrix2 = self.env.view_matrix
        proj_matrix2 = self.env.projection_matrix
        
        # 2. Convert to Point Clouds
        pc1 = depth_to_point_cloud(depth1, view_matrix1, proj_matrix1, self.env.camera_width, self.env.camera_height)
        pc2 = depth_to_point_cloud(depth2, view_matrix2, proj_matrix2, self.env.camera_width, self.env.camera_height)
        
        # Combine
        full_pc = np.concatenate([pc1, pc2], axis=0)
        
        # 3. Filter workspace (optional clean up)
        # Remove floor (z < 0.01) and too high (z > 1.5)
        mask = (full_pc[:, 2] > 0.01) & (full_pc[:, 2] < 1.5)
        # Also limit XY workspace if needed
        mask &= (np.abs(full_pc[:, 0]) < 1.0) & (np.abs(full_pc[:, 1]) < 1.0)
        
        filtered_pc = full_pc[mask]
        
        # 4. Downsample to fixed size
        if len(filtered_pc) == 0:
             final_pc = np.zeros((self.num_points, 3))
        else:
             final_pc = fps_downsample(filtered_pc, self.num_points)
             
        # 5. Gripper Keypoints
        gripper_pcd = self.env.robot.get_gripper_keypoints()
        
        return {
            'point_cloud': final_pc.astype(np.float32),
            'agent_pos': obs.astype(np.float32),
            'gripper_pcd': gripper_pcd.astype(np.float32)
        }
