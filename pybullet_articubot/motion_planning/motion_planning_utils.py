"""
High-level motion planning utilities.
Replicates ArticuBot's manipulation/motion_planning_utils.py.
"""

import numpy as np
import pybullet as p
import copy
from termcolor import cprint
from motion_planning import pb_ompl

def get_path_length(env, path):
    """
    Calculate path translation and rotation length.
    """
    cur_pos, cur_orient = env.robot.get_end_effector_pose()
    length_pos = 0
    length_orient = 0
    
    for q in path:
        env.robot.set_joint_angles(env.robot.arm_joint_indices, q)
        pos, orient = env.robot.get_end_effector_pose()
        length_pos += np.linalg.norm(pos - cur_pos)
        
        # Orientation distance (simplified)
        # ArticuBot uses dot product based distance
        dot = np.dot(orient, cur_orient)
        temp = 2 * dot**2 - 1
        temp = np.clip(temp, -1, 1)
        length_orient += np.arccos(temp)
        
        cur_pos, cur_orient = pos, orient
        
    return length_pos, length_orient

def motion_planning(env, target_pos, target_orientation, 
                    obstacles=[], allow_collision_links=[], object_id=None,
                    max_sampling_it=80, smooth_path=True, try_times=3):
    """
    Plan motion to target pose using OMPL.
    
    Args:
        env: Simulation environment
        target_pos: Target EE position
        target_orientation: Target EE orientation
        obstacles: List of obstacle IDs
        object_id: ID of held object (if any)
        
    Returns:
        (success, path, translation_len, rotation_len)
    """
    current_joint_angles = env.robot.get_arm_joint_angles()
    
    # --- PLANNER SELECTION ---
    if pb_ompl.HAS_OMPL:
        # Use OMPL Planners
        ompl_robot = pb_ompl.PbOMPLRobot(env.robot.body, 
                                         control_joint_idx=env.robot.arm_joint_indices,
                                         object_id=object_id, env=env)
        ompl_robot.set_state(current_joint_angles)
        
        pb_ompl_interface = pb_ompl.PbOMPL(ompl_robot, obstacles, allow_collision_links, object_id=object_id)
        
        for try_idx in range(try_times):
            solutions = env.robot.ik_with_random_restart(target_pos, target_orientation, 
                                                         env.robot.end_effector, 
                                                         env.robot.arm_joint_indices,
                                                         num_attempts=max_sampling_it)
            valid_solutions = [sol for sol in solutions if pb_ompl_interface.is_state_valid(sol)]
            
            if not valid_solutions:
                continue
                
            curr = np.array(current_joint_angles)
            best_sol = valid_solutions[np.argmin([np.linalg.norm(np.array(s) - curr) for s in valid_solutions])]
            
            for planner in ["RRTConnect", "BITstar"]:
                pb_ompl_interface.set_planner(planner)
                ompl_robot.set_state(current_joint_angles)
                res, path = pb_ompl_interface.plan(best_sol, smooth_path=smooth_path)
                if res:
                    trans_len, rot_len = get_path_length(env, path)
                    env.robot.set_joint_angles(env.robot.arm_joint_indices, current_joint_angles)
                    return True, path, trans_len, rot_len
    else:
        # --- FALLBACK: Pure Python RRTConnect ---
        from .rrt_planner import RRTConnect as PyRRTConnect
        planner = PyRRTConnect(env.robot.body, env.robot.arm_joint_indices, obstacles, physics_id=env.id)
        
        for try_idx in range(try_times):
            solutions = env.robot.ik_with_random_restart(target_pos, target_orientation, 
                                                         env.robot.end_effector, 
                                                         env.robot.arm_joint_indices,
                                                         num_attempts=max_sampling_it)
            valid_solutions = [sol for sol in solutions if planner.is_collision_free(sol)]
            if not valid_solutions: continue
            
            curr = np.array(current_joint_angles)
            best_sol = valid_solutions[np.argmin([np.linalg.norm(np.array(s) - curr) for s in valid_solutions])]
            
            path = planner.plan(current_joint_angles, best_sol)
            if path:
                if smooth_path:
                    path = planner.smooth_path(path)
                trans_len, rot_len = get_path_length(env, path)
                env.robot.set_joint_angles(env.robot.arm_joint_indices, current_joint_angles)
                return True, path, trans_len, rot_len
                
    env.robot.set_joint_angles(env.robot.arm_joint_indices, current_joint_angles)
    return False, None, None, None
