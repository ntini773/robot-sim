"""
SimpleEnv module.
Core simulation environment for ArticuBot-style UR5 setup.
"""

import os
import time
import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
from typing import Optional, Dict, List, Tuple, Any

from ur5 import UR5Robotiq85
from utils.config_parser import parse_config
from utils.camera import setup_camera_rpy, render
from utils.state_io import save_env, load_env

class SimpleEnv(gym.Env):
    """
    Gym-compatible environment for UR5 manipulation.
    Replicates ArticuBot's SimpleEnv functionality.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 gui: bool = False,
                 dt: float = 1/240,
                 control_step: int = 5,
                 horizon: int = 250,
                 randomize: int = 0,
                 robot_base_pos: List[float] = [0, -0.6, 0]):
        
        super().__init__()
        
        self.config_path = config_path
        self.gui = gui
        self.dt = dt
        self.control_step = control_step
        self.horizon = horizon
        self.randomize = randomize
        self.robot_base_pos = robot_base_pos
        
        # PyBullet initialization
        if self.gui:
            self.id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1, physicsClientId=self.id)
        else:
            self.id = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setTimeStep(self.dt, physicsClientId=self.id)
        
        # Action/Observation Spaces
        self._setup_spaces()
        
        # Environment State
        self.robot: Optional[UR5Robotiq85] = None
        self.urdf_ids: Dict[str, int] = {}
        self.urdf_paths: List[str] = []
        self.simulator_sizes: Dict[str, float] = {}
        self.time_step: int = 0
        
        # Camera configuration
        self.camera_width = 640
        self.camera_height = 480
        self.view_matrix = None
        self.projection_matrix = None
        
        # Initial call to set scene structure (without full reset)
        if self.config_path:
            self.set_scene()
            
    def _setup_spaces(self):
        """Define action and observation spaces."""
        # Action: 3 pos + 3 euler + 1 gripper
        self.action_space = spaces.Box(
            low=np.array([-1]*7), 
            high=np.array([1]*7), 
            dtype=np.float32
        )
        
        # Observation - placeholder, will be overridden by wrapper usually
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,), 
            dtype=np.float32
        )
        
    def set_scene(self, reset_state: Optional[Dict] = None):
        """
        Set up the scene, load robot and objects.
        """
        p.resetSimulation(physicsClientId=self.id)
        
        # Reset physics config
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setRealTimeSimulation(0, physicsClientId=self.id) # Exact simulation
        
        # Load Plane
        urdf_dir = os.path.dirname(os.path.abspath(__file__))
        plane_path = os.path.join(urdf_dir, "assets", "plane", "plane.urdf")
        
        try:
             plane_id = p.loadURDF(plane_path, physicsClientId=self.id)
        except:
             plane_id = p.loadURDF("plane.urdf", physicsClientId=self.id) # PyBullet built-in
             
        self.urdf_ids['plane'] = plane_id
        
        # Load Robot
        self.robot = UR5Robotiq85(self.id)
        self.robot.load(self.robot_base_pos, fixed_base=True)
        self.urdf_ids['robot'] = self.robot.body
        
        # Load Objects from Config
        if self.config_path:
            self._load_objects_from_config()
            
        # Initial stabilization
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)
            
        # Restore state if provided
        if reset_state:
            load_env(self, state=reset_state)
            
    def _load_objects_from_config(self):
        """Parse config and load URDFs."""
        (urdf_paths, sizes, positions, orientations, 
         names, types, on_tables, use_table) = parse_config(self.config_path)
        
        self.urdf_paths = urdf_paths
        
        for i, path in enumerate(urdf_paths):
            name = names[i] if i < len(names) else f"obj_{i}"
            pos = positions[i]
            orn_euler = orientations[i]
            if len(orn_euler) == 3:
                 orn = list(p.getQuaternionFromEuler(orn_euler))
            else:
                 orn = list(orn_euler)
            
            size = sizes[i]
            
            # Use Fixed base for articulated objects to prevent falling over
            # unless explicitly specified otherwise (omitted complexity for now)
            use_fixed = True 
            
            # Load object
            try:
                if not os.path.isabs(path) and not os.path.exists(path):
                    # Try relative to the package root
                    root_dir = os.path.dirname(os.path.abspath(__file__))
                    alt_path = os.path.join(root_dir, path)
                    if os.path.exists(alt_path):
                        path = alt_path

                obj_id = p.loadURDF(
                    path,
                    basePosition=pos,
                    baseOrientation=orn,
                    useFixedBase=use_fixed,
                    globalScaling=size,
                    physicsClientId=self.id
                )
                self.urdf_ids[name] = obj_id
                self.simulator_sizes[name] = size
                
                # Set friction
                num_joints = p.getNumJoints(obj_id, physicsClientId=self.id)
                for j in range(num_joints):
                    p.changeDynamics(obj_id, j, lateralFriction=5.0, spinningFriction=5.0, physicsClientId=self.id)
                    
            except Exception as e:
                print(f"Failed to load object {name} from {path}: {e}")

    def reset(self):
        """
        Reset environment.
        """
        self.time_step = 0
        self.set_scene()
        self.robot.reset_to_rest_pose()
        self.robot.open_gripper()
        
        # Stabilize
        for _ in range(200):
            p.stepSimulation(physicsClientId=self.id)
            
        return self._get_obs(), {}
        
    def step(self, action):
        """
        Execute action and step simulation.
        Action: [x, y, z, roll, pitch, yaw, gripper]
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Parse action (relative or absolute - sticking to Absolute for simplicity as ArticuBot uses IK)
        # ArticuBot `take_direct_action` takes target pose
        
        target_pos = action[:3]
        target_orn_euler = action[3:6]
        gripper_val = action[6]
        
        # ArticuBot often uses delta action or absolute. Let's assume Absolute for IK interface
        target_orn_quat = p.getQuaternionFromEuler(target_orn_euler)
        
        # Gripper
        if gripper_val > 0.5: # Simple threshold
            self.robot.close_gripper()
        else:
            self.robot.open_gripper()
            
        # Arm movement
        # Try finding IK solution
        self.robot.move_to_pose(target_pos, target_orn_quat)
        
        # Step simulation multiple times
        for _ in range(self.control_step):
            p.stepSimulation(physicsClientId=self.id)
            if self.gui:
                time.sleep(self.dt)
                
        self.time_step += 1
        done = self.time_step >= self.horizon
        
        return self._get_obs(), 0.0, done, {}
        
    def _get_obs(self):
        """
        Get observation.
        Placeholder, wrapped by PointCloudWrapper.
        """
        return self.robot.get_state()
        
    def render(self, mode='rgb_array'):
        """
        Render scene.
        """
        if self.view_matrix is None:
             self.setup_camera(
                 camera_target=[0, 0, 0],
                 distance=1.5,
                 yaw=45, pitch=-30
             )
             
        width, height = self.camera_width, self.camera_height
        rgb, depth, seg = render(width, height, self.view_matrix, self.projection_matrix, self.id)
        
        return rgb, depth, seg
        
    def setup_camera(self, camera_target, distance, yaw, pitch, roll=0):
        """Setup main camera."""
        self.view_matrix = setup_camera_rpy(camera_target, distance, yaw, pitch, roll)
        fov = 60
        aspect = self.camera_width / self.camera_height
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, 0.01, 100, physicsClientId=self.id)

    def close(self):
        p.disconnect(self.id)

