"""
Pure Python RRT-Connect Planner.
Fallback for when OMPL is not available.
Adapted from the user's master-cleaner-position.py.
"""

import numpy as np
import pybullet as p
import random
from typing import List, Optional, Tuple

from motion_planning import collision_utils as pb_utils

class RRTConnect:
    def __init__(self, robot_id, joint_indices, obstacles=[], step_size=0.15, max_iters=5000, physics_id=0):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iters = max_iters
        self.physics_id = physics_id
        
        # Get joint limits
        self.lower_limits = []
        self.upper_limits = []
        for j in joint_indices:
            info = p.getJointInfo(robot_id, j, physicsClientId=physics_id)
            self.lower_limits.append(info[8])
            self.upper_limits.append(info[9])
        self.lower_limits = np.array(self.lower_limits)
        self.upper_limits = np.array(self.upper_limits)

    def is_collision_free(self, q):
        """Check if joint configuration q is collision-free."""
        # Store current state
        current_q = [p.getJointState(self.robot_id, i, physicsClientId=self.physics_id)[0] for i in self.joint_indices]
        
        # Reset joints to q for check
        for i, joint_id in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_id, q[i], physicsClientId=self.physics_id)
        
        p.performCollisionDetection(physicsClientId=self.physics_id)
        
        # Use existing collision utilities for robust checking
        # Check self collisions
        # We need self link pairs
        self_collision = False
        # Simplified self-collision check for speed if needed, but let's be thorough
        # For now, check if robot is in collision with any obstacle
        for obs_id in self.obstacles:
            if pb_utils.pairwise_collision(self.robot_id, obs_id, p_id=self.physics_id):
                self_collision = True
                break
        
        # Restore
        for i, joint_id in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_id, current_q[i], physicsClientId=self.physics_id)
            
        return not self_collision

    def sample_q(self):
        return np.random.uniform(self.lower_limits, self.upper_limits)

    def plan(self, q_start, q_goal):
        if not self.is_collision_free(q_start):
            return None
        if not self.is_collision_free(q_goal):
            return None

        tree_start = {tuple(q_start): None}
        tree_goal = {tuple(q_goal): None}
        
        tree_a, tree_b = tree_start, tree_goal
        a_is_start = True
        
        path = None
        for _ in range(self.max_iters):
            q_rand = self.sample_q()
            q_near_a = self.get_nearest(tree_a, q_rand)
            q_new_a = self.step_towards(q_near_a, q_rand)
            
            if q_new_a is not None and self.is_collision_free(q_new_a):
                tree_a[tuple(q_new_a)] = tuple(q_near_a)
                
                # Try to connect Tree B
                q_near_b = self.get_nearest(tree_b, q_new_a)
                curr_q_b = q_near_b
                while True:
                    q_new_b = self.step_towards(curr_q_b, q_new_a)
                    if q_new_b is None or not self.is_collision_free(q_new_b):
                        break
                    tree_b[tuple(q_new_b)] = tuple(curr_q_b)
                    curr_q_b = q_new_b
                    
                    if np.linalg.norm(np.array(curr_q_b) - np.array(q_new_a)) < self.step_size:
                        if a_is_start:
                            path = self.extract_ordered_path(tree_start, q_new_a, tree_goal, curr_q_b)
                        else:
                            path = self.extract_ordered_path(tree_start, curr_q_b, tree_goal, q_new_a)
                        break
                if path: break

            tree_a, tree_b = tree_b, tree_a
            a_is_start = not a_is_start

        return path

    def extract_ordered_path(self, tree_start, q_s, tree_goal, q_g):
        path_start = []
        curr = tuple(q_s)
        while curr is not None:
            path_start.append(curr)
            curr = tree_start[curr]
        path_start.reverse()
        
        path_goal = []
        curr = tuple(q_g)
        while curr is not None:
            path_goal.append(curr)
            curr = tree_goal[curr]
            
        return path_start + path_goal

    def get_nearest(self, tree, q_target):
        nodes = np.array(list(tree.keys()))
        distances = np.linalg.norm(nodes - q_target, axis=1)
        return nodes[np.argmin(distances)]

    def step_towards(self, q_from, q_to):
        diff = np.array(q_to) - np.array(q_from)
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return None
        step = diff * min(self.step_size / dist, 1.0)
        return np.array(q_from) + step

    def smooth_path(self, path, max_iters=50):
        if path is None or len(path) < 3:
            return path
        
        smoothed = list(path)
        for _ in range(max_iters):
            if len(smoothed) < 3: break
            i = random.randint(0, len(smoothed) - 3)
            j = random.randint(i + 2, len(smoothed) - 1)
            
            if self.is_path_collision_free(smoothed[i], smoothed[j]):
                smoothed = smoothed[:i+1] + smoothed[j:]
        return smoothed

    def is_path_collision_free(self, q1, q2, steps=10):
        for i in range(1, steps):
            q = np.array(q1) + (np.array(q2) - np.array(q1)) * (i / steps)
            if not self.is_collision_free(q):
                return False
        return True
