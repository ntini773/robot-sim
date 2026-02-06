"""
Demo script to verify ArticuBot-style UR5 setup.
"""

import sys
import os
import time
import numpy as np
import pybullet as p
from termcolor import cprint

# Ensure package is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sim import SimpleEnv
from wrapper import PointCloudWrapper
from motion_planning.motion_planning_utils import motion_planning

def main():
    # Initialize Environment
    print("Initializing environment...")
    config_path = os.path.join(os.path.dirname(__file__), "configs", "example_scene.yaml")
    
    env = SimpleEnv(
        config_path=config_path,
        gui=True,
        robot_base_pos=[0, -0.6, 0]
    )
    
    # Wrap with Point Cloud Wrapper
    env = PointCloudWrapper(env, num_points=1000)
    
    # Reset
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Point cloud shape: {obs['point_cloud'].shape}")
    print(f"Agent State: {obs['agent_pos']}")
    
    # Move to a target pose using Motion Planning
    print("\n--- Testing Motion Planning ---")
    target_pos = [0.2, -0.4, 0.4]
    target_orn = p.getQuaternionFromEuler([0, np.pi, 0]) # Downward facing
    
    cprint(f"Planning motion to {target_pos}...", "cyan")
    success, path, t_len, r_len = motion_planning(
        env.env, target_pos, target_orn,
        try_times=3, max_sampling_it=50
    )
    
    if success:
        cprint(f"Planning SUCCESS! Path length: {len(path)}", "green")
        
        # Execute Path
        for q in path:
            env.env.robot.control(env.env.robot.arm_joint_indices, q)
            # Stabilize
            for _ in range(5):
                 p.stepSimulation(physicsClientId=env.env.id)
                 time.sleep(0.01)
            
        cprint("Goal Reached!", "green")
        
        # Test Gripper
        cprint("\n--- Testing Gripper ---", "cyan")
        env.env.robot.close_gripper()
        for _ in range(50):
             p.stepSimulation(physicsClientId=env.env.id)
             time.sleep(0.01)
             
        env.env.robot.open_gripper()
        for _ in range(50):
             p.stepSimulation(physicsClientId=env.env.id)
             time.sleep(0.01)
             
    else:
        cprint("Planning FAILED!", "red")
        
    # Get final observation
    obs = env.observation(env.env.robot.get_state())
    cprint("\nFinal Point Cloud Stats:", "cyan")
    print(f"Points: {len(obs['point_cloud'])}")
    print(f"Min Bounds: {np.min(obs['point_cloud'], axis=0)}")
    print(f"Max Bounds: {np.max(obs['point_cloud'], axis=0)}")
    
    time.sleep(2)
    # env.close()

if __name__ == "__main__":
    main()
