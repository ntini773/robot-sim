"""
Collision checking utilities for OMPL interface.
Replicates ArticuBot's pybullet_ompl/utils.py collision logic.
"""

from __future__ import print_function
import pybullet as p
from itertools import product, combinations
from collections import namedtuple

BASE_LINK = -1
MAX_DISTANCE = 0.0

def pairwise_link_collision(body1, link1, body2, link2=BASE_LINK, max_distance=MAX_DISTANCE, p_id=None):
    """Check collision between two specific links."""
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  linkIndexA=link1, linkIndexB=link2, physicsClientId=p_id)) != 0

def pairwise_collision(body1, body2, p_id=None):
    """Check collision between two bodies (checking all links)."""
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=MAX_DISTANCE, physicsClientId=p_id)) != 0

def get_self_link_pairs(body, joints, disabled_collisions=set(), only_moving=True):
    """Get list of link pairs to check for self-collision."""
    moving_links = get_moving_links(body, joints)
    fixed_links = list(set(get_joints(body)) - set(moving_links))
    check_link_pairs = list(product(moving_links, fixed_links))
    
    if only_moving:
        check_link_pairs.extend(get_moving_pairs(body, joints))
    else:
        check_link_pairs.extend(combinations(moving_links, 2))
        
    check_link_pairs = list(
        filter(lambda pair: not are_links_adjacent(body, *pair), check_link_pairs))
    check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                (pair[::-1] not in disabled_collisions), check_link_pairs))
    return check_link_pairs

def get_moving_links(body, joints):
    """Get all links that move when specified joints move."""
    moving_links = set()
    for joint in joints:
        link = child_link_from_joint(joint)
        if link not in moving_links:
            moving_links.update(get_link_subtree(body, link))
    return list(moving_links)

def get_moving_pairs(body, moving_joints):
    """Get all pairs of moving links."""
    moving_links = get_moving_links(body, moving_joints)
    for link1, link2 in combinations(moving_links, 2):
        ancestors1 = set(get_joint_ancestors(body, link1)) & set(moving_joints)
        ancestors2 = set(get_joint_ancestors(body, link2)) & set(moving_joints)
        if ancestors1 != ancestors2:
            yield link1, link2

# Helper functions
def get_num_joints(body):
    return p.getNumJoints(body)

def get_joints(body):
    return list(range(get_num_joints(body)))

def get_joint_info(body, joint):
    return p.getJointInfo(body, joint)

def child_link_from_joint(joint):
    return joint

def get_link_parent(body, link):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link)[16] # parentIndex is at index 16

def get_link_ancestors(body, link):
    parent = get_link_parent(body, link)
    if parent is None:
        return []
    return get_link_ancestors(body, parent) + [parent]

def get_joint_ancestors(body, joint):
    link = child_link_from_joint(joint)
    return get_link_ancestors(body, link) + [link]

def get_link_children(body, link):
    children = []
    for i in range(get_num_joints(body)):
        if get_link_parent(body, i) == link:
            children.append(i)
    return children

def get_link_descendants(body, link, test=lambda l: True):
    descendants = []
    for child in get_link_children(body, link):
        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(body, child, test=test))
    return descendants

def get_link_subtree(body, link, **kwargs):
    return [link] + get_link_descendants(body, link, **kwargs)

def are_links_adjacent(body, link1, link2):
    return (get_link_parent(body, link1) == link2) or \
           (get_link_parent(body, link2) == link1)
