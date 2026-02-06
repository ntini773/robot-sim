"""
State I/O utilities.
Handles saving and loading of environment state.
"""

import pickle
import json
import numpy as np
import os
import pybullet as p
from typing import Dict, Any, List

def save_env(env, save_path: str = None) -> Dict[str, Any]:
    """
    Save the current state of the environment.
    
    Args:
        env: The environment instance
        save_path: Optional path to save the pickle file
        
    Returns:
        State dictionary
    """
    state = {
        'robot_joint_angles': env.robot.get_joint_angles(),
        'robot_base_pos': env.robot.get_pos_orient(-1)[0],
        'robot_base_orn': env.robot.get_pos_orient(-1)[1],
        'object_states': {}
    }
    
    # Save object states
    for name, obj_id in env.urdf_ids.items():
        if name == 'robot' or name == 'plane':
            continue
            
        pos, orn = p.getBasePositionAndOrientation(obj_id, physicsClientId=env.id)
        
        joint_states = []
        num_joints = p.getNumJoints(obj_id, physicsClientId=env.id)
        for i in range(num_joints):
            joint_states.append(p.getJointState(obj_id, i, physicsClientId=env.id)[0])
            
        state['object_states'][name] = {
            'pos': pos,
            'orn': orn,
            'joint_angles': joint_states
        }
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
            
    return state

def load_env(env, load_path: str = None, state: Dict[str, Any] = None):
    """
    Load environment state.
    
    Args:
        env: The environment instance
        load_path: Path to pickle file
        state: Direct state dictionary (if load_path not provided)
    """
    if load_path:
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
            
    if state is None:
        return
    
    # Restore robot
    if 'robot_joint_angles' in state:
        env.robot.set_joint_angles(env.robot.controllable_joints, state['robot_joint_angles'])
    
    # Restore objects
    if 'object_states' in state:
        for name, obj_state in state['object_states'].items():
            if name in env.urdf_ids:
                obj_id = env.urdf_ids[name]
                p.resetBasePositionAndOrientation(obj_id, obj_state['pos'], obj_state['orn'], physicsClientId=env.id)
                
                for i, angle in enumerate(obj_state['joint_angles']):
                    p.resetJointState(obj_id, i, angle, physicsClientId=env.id)

def save_trajectory(save_path: str, trajectory: List[Dict[str, Any]]):
    """
    Save a trajectory of observations/actions.
    
    Args:
        save_path: Path to save file (.pkl or .npz)
        trajectory: List of steps
    """
    with open(save_path, 'wb') as f:
        pickle.dump(trajectory, f)
