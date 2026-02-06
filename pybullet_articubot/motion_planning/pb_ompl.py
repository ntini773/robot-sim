"""
OMPL-PyBullet Interface.
Replicates ArticuBot's pybullet_ompl/pb_ompl.py logic.
Handles OMPL setup, state validity checking, and planning.
"""

HAS_OMPL = False
try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
    ou.setLogLevel(ou.LOG_ERROR)
    HAS_OMPL = True
except ImportError:
    # Fallback or error if OMPL not installed
    # Define mock objects to allow parsing
    class Mock:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return Mock
    ob = Mock()
    og = Mock()
    ou = Mock()

import pybullet as p
import time
import copy
import numpy as np
import random
from itertools import product
from motion_planning import collision_utils as pb_utils

INTERPOLATE_NUM = 100
DEFAULT_PLANNING_TIME = 5.0

class PbOMPLRobot:
    """Wrapper for robot to be used with OMPL."""
    def __init__(self, body_id, control_joint_idx=None, object_id=None, env=None):
        self.id = body_id
        self.env = env
        self.physics_id = env.id if env else 0
        
        # Determine controllable joints
        all_joint_num = p.getNumJoints(body_id, physicsClientId=self.physics_id)
        if control_joint_idx is None:
            self.joint_idx = [j for j in range(all_joint_num) if self._is_not_fixed(j)]
        else:
            self.joint_idx = control_joint_idx
            
        self.num_dim = len(self.joint_idx)
        self.joint_bounds = []
        self.get_joint_bounds()
        
        self.object_id = object_id
        
        # Calculate transform for held object
        if object_id is not None:
             object_pos, object_orn = p.getBasePositionAndOrientation(object_id, physicsClientId=self.physics_id)
             robot_pos, robot_orn = env.robot.get_end_effector_pose()
             
             world_to_robot = p.invertTransform(robot_pos, robot_orn)
             object_in_robot = p.multiplyTransforms(world_to_robot[0], world_to_robot[1], object_pos, object_orn)
             self.object_in_robot_pos, self.object_in_robot_orn = object_in_robot
        
    def _is_not_fixed(self, joint_idx):
        info = p.getJointInfo(self.id, joint_idx, physicsClientId=self.physics_id)
        return info[2] != p.JOINT_FIXED
        
    def get_joint_bounds(self):
        for i, joint_id in enumerate(self.joint_idx):
            info = p.getJointInfo(self.id, joint_id, physicsClientId=self.physics_id)
            low, high = info[8], info[9]
            
            # Reduce range slightly to avoid singularities/limits
            if low < high:
                 delta = 0.05 * (high - low)
                 low += delta
                 high -= delta
                 self.joint_bounds.append([low, high])
        return self.joint_bounds
        
    def get_cur_state(self):
        return [p.getJointState(self.id, i, physicsClientId=self.physics_id)[0] for i in self.joint_idx]
        
    def set_state(self, state):
        for joint, value in zip(self.joint_idx, state):
            p.resetJointState(self.id, joint, value, targetVelocity=0, physicsClientId=self.physics_id)
            
        if self.object_id is not None and self.env:
             robot_pos, robot_orn = self.env.robot.get_end_effector_pose()
             obj_pos, obj_orn = p.multiplyTransforms(robot_pos, robot_orn, 
                                                     self.object_in_robot_pos, self.object_in_robot_orn)
             p.resetBasePositionAndOrientation(self.object_id, obj_pos, obj_orn, physicsClientId=self.physics_id)

class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim):
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None
        
    def allocStateSampler(self):
        if self.state_sampler:
            return self.state_sampler
        return self.allocDefaultStateSampler()
        
    def set_state_sampler(self, state_sampler):
        self.state_sampler = state_sampler

class PbOMPL:
    def __init__(self, robot, obstacles=[], allow_collision_links=[], 
                 allow_collision_robot_link_pairs=[], object_id=None,
                 interpolation_num=INTERPOLATE_NUM):
        
        self.robot = robot
        self.robot_id = robot.id
        self.obstacles = obstacles
        self.allow_collision_links = allow_collision_links
        self.allow_collision_robot_link_pairs = allow_collision_robot_link_pairs
        self.interpolation_num = interpolation_num
        self.object_id = object_id
        
        # OMPL setup
        self.space = PbStateSpace(robot.num_dim)
        
        bounds = ob.RealVectorBounds(robot.num_dim)
        for i, bound in enumerate(robot.joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.space.setBounds(bounds)
        
        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()
        
        self.setup_collision_detection(robot, obstacles)
        self.set_object(object_id)
        self.set_planner("RRTConnect")
        
    def setup_collision_detection(self, robot, obstacles, self_collisions=True):
        self.check_link_pairs = pb_utils.get_self_link_pairs(robot.id, robot.joint_idx) if self_collisions else []
        for pair in self.allow_collision_robot_link_pairs:
            if pair in self.check_link_pairs:
                self.check_link_pairs.remove(pair)
                
        moving_links = frozenset([
            item for item in pb_utils.get_moving_links(robot.id, robot.joint_idx)
            if item not in self.allow_collision_links
        ])
        moving_bodies = [(robot.id, moving_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))
        
    def set_object(self, object_id):
        self.object_id = object_id
        if object_id is not None:
            self.check_body_pairs += list(product([(object_id, None)], self.obstacles))
            
    def set_planner(self, planner_name):
        if planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.si)
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.si)
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.si)
        elif planner_name == "ABITstar":
            self.planner = og.ABITstar(self.si)
        else:
            print(f"Planner {planner_name} not recognized, defaulting to RRTConnect")
            self.planner = og.RRTConnect(self.si)
        self.ss.setPlanner(self.planner)
        
    def is_state_valid(self, state):
        # OMPL state to list
        state_list = [state[i] for i in range(self.robot.num_dim)]
        self.robot.set_state(state_list)
        
        # Check self collision
        for link1, link2 in self.check_link_pairs:
            if pb_utils.pairwise_link_collision(self.robot_id, link1, self.robot_id, link2, p_id=self.robot.physics_id):
                return False
                
        # Check collision with environment
        for body_info, obstacle_id in self.check_body_pairs:
             body_id = body_info[0]
             # If we had fine-grained link info for body1, we could check specific links
             # But pairwise_collision checks all links
             if pb_utils.pairwise_collision(body_id, obstacle_id, p_id=self.robot.physics_id):
                 return False
                 
        return True
        
    def plan(self, goal, allowed_time=DEFAULT_PLANNING_TIME, smooth_path=True):
        start = self.robot.get_cur_state()
        
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]
            
        self.ss.setStartAndGoalStates(s, g)
        
        solved = self.ss.solve(allowed_time)
        res = False
        path_list = []
        
        if solved:
            res = True
            path = self.ss.getSolutionPath()
            if smooth_path:
                ps = og.PathSimplifier(self.si)
                ps.simplify(path, allowed_time)
                
            path.interpolate(self.interpolation_num)
            
            states = path.getStates()
            path_list = [[states[i][j] for j in range(self.robot.num_dim)] 
                         for i in range(len(states))]
                         
        return res, path_list
