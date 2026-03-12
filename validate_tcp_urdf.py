import pybullet as p
import numpy as np
import time 
def link_name_to_index(robot, link_name):
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        name = info[12].decode("utf-8")
        if name == link_name:
            return i
    raise ValueError(f"Link {link_name} not found")

def draw_frame(body, link_name, axis_len=0.1):

    # find link index
    link_index = None
    for i in range(p.getNumJoints(body)):
        info = p.getJointInfo(body, i)
        name = info[12].decode("utf-8")
        if name == link_name:
            link_index = i
            break

    if link_index is None:
        raise ValueError("Link not found")

    state = p.getLinkState(body, link_index)
    pos = np.array(state[4])
    orn = state[5]

    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)

    x_axis = pos + axis_len * rot[:,0]
    y_axis = pos + axis_len * rot[:,1]
    z_axis = pos + axis_len * rot[:,2]

    p.addUserDebugLine(pos, x_axis, [1,0,0], 3)  # X (red)
    p.addUserDebugLine(pos, y_axis, [0,1,0], 3)  # Y (green)
    p.addUserDebugLine(pos, z_axis, [0,0,1], 3)  # Z (blue)

if __name__ == "__main__":
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    robot_id = p.loadURDF("./lite-6-updated-urdf/lite_6_new.urdf", [0,0,0.62], useFixedBase=True)
    # get link indices
    base_idx = link_name_to_index(robot_id, "base_link")
    left_idx = link_name_to_index(robot_id, "left_finger")
    right_idx = link_name_to_index(robot_id, "right_finger")

    base_pos = np.array(p.getLinkState(robot_id, base_idx)[4])
    left_pos = np.array(p.getLinkState(robot_id, left_idx)[4])
    right_pos = np.array(p.getLinkState(robot_id, right_idx)[4])

    midpoint = (left_pos + right_pos) / 2

    offset = midpoint - base_pos
    print("TCP offset from base_link:", offset)
    draw_frame(robot_id, "tcp", axis_len=0.2)
    draw_frame(robot_id, "base_link", axis_len=0.2)
    while True:
        p.stepSimulation()
        time.sleep(1/240)