import pybullet as p
import pybullet_data
import numpy as np
from pybullet_planning import (
    load_pybullet,
    get_movable_joints,
    set_joint_positions,
    connect, plan_joint_motion
)
from pybullet_planning import get_collision_fn

import time 

class PandaRobot:

    def __init__(self, base_pos=[0,0,0], base_ori=[0,0,0]):
        self.base_pos = base_pos
        self.base_ori = p.getQuaternionFromEuler(base_ori)

        self.robot_id = None
        self.joints = None
        self.dof = None
        self.lower_limits = None
        self.upper_limits = None

    # ---------------------------------
    # Load robot
    # ---------------------------------
    def load(self):

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.robot_id = load_pybullet(
            "franka_panda/panda.urdf",
            # "lite-6-updated-urdf/lite_6_new.urdf",
            fixed_base=True
        )

        p.resetBasePositionAndOrientation(
            self.robot_id,
            self.base_pos,
            self.base_ori
        )

        self._parse_joints()


    # ---------------------------------
    # Parse joints automatically
    # ---------------------------------
    def _parse_joints(self):

        joints = get_movable_joints(self.robot_id)

        # Panda has 7 arm joints
        self.joints = joints[:6]

        self.dof = len(self.joints)

        lower = []
        upper = []

        for j in self.joints:

            info = p.getJointInfo(self.robot_id, j)

            lower.append(info[8])
            upper.append(info[9])

        self.lower_limits = np.array(lower)
        self.upper_limits = np.array(upper)


    # ---------------------------------
    # Robot API (planner will use this)
    # ---------------------------------

    def get_dof(self):
        return self.dof


    def get_joint_limits(self):
        return self.lower_limits, self.upper_limits


    def get_joint_positions(self):

        states = p.getJointStates(self.robot_id, self.joints)

        return np.array([s[0] for s in states])


    def set_joint_positions(self, q):

        set_joint_positions(
            self.robot_id,
            self.joints,
            q
        )


    def get_robot_id(self):
        return self.robot_id


    def get_joints(self):
        return self.joints


    # ---------------------------------
    # Debug utility
    # ---------------------------------

    def print_info(self):

        print("Robot ID:", self.robot_id)
        print("DOF:", self.dof)
        print("Joints:", self.joints)
        print("Lower:", self.lower_limits)
        print("Upper:", self.upper_limits)

def create_box(position, size=[0.1,0.1,0.1], color=[1,0,0,1]):

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=size
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=size,
        rgbaColor=color
    )

    body = p.createMultiBody(
        baseMass=0,                 # <-- IMPORTANT
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )

    return body

def print_robot_collisions(robot):
    contacts = p.getContactPoints(bodyA=robot)

    if len(contacts) == 0:
        print("No collisions detected")
        return

    print(f"\nDetected {len(contacts)} collisions:\n")

    for c in contacts:
        bodyA = c[1]
        bodyB = c[2]
        linkA = c[3]
        linkB = c[4]
        distance = c[8]

        nameA = p.getBodyInfo(bodyA)[1].decode()
        nameB = p.getBodyInfo(bodyB)[1].decode()

        print(
            f"{nameA} (link {linkA})  <-->  {nameB} (link {linkB}) "
            f"| penetration: {distance:.4f}"
        )
def setup_environment():

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    
    plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useMaximalCoordinates=True)
    table_id = p.loadURDF(
        "table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0])
    )
    p.setTimeStep(1/240)
    # floating obstacle
    box1 = create_box(
        position=[0.5,0,0.9],
        size=[0.1,0.08,0.1],
        color=[1,0,0,1]
    )

    # box2 = create_box(
    #     position=[0.45,0.2,0.8],
    #     size=[0.05,0.05,0.15],
    #     color=[0,0,1,1]
    # )
    obstacles = [plane_id, table_id,box1]

    return table_id, obstacles

def draw_sphere(position, color=[1,0,0], radius=0.03):

    visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color + [1]
    )

    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=position
    )

def move_to_waypoint(robot_id, joints, q, steps=400, dt=1/240):
    """Drive joints to waypoint q, stepping simulation until converged."""
    for _ in range(steps):
        for i, joint_id in enumerate(joints):
            p.setJointMotorControl2(
                robot_id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=q[i],
                maxVelocity=3.0,
                force=200
            )
        p.stepSimulation()
        time.sleep(dt)

        # Early exit if all joints close enough to target
        states = p.getJointStates(robot_id, joints)
        current = np.array([s[0] for s in states])
        if np.max(np.abs(current - np.array(q))) < 0.01:
            break
def solve_ik_collision_free(robot_id, joints, pos, orn, collision_fn, ee_link=11, max_attempts=50):
    """Try multiple IK seeds until a collision-free solution is found."""
    lower = np.array([p.getJointInfo(robot_id, j)[8] for j in joints])
    upper = np.array([p.getJointInfo(robot_id, j)[9] for j in joints])

    for attempt in range(max_attempts):
        # Use random seed after first attempt
        if attempt > 0:
            seed = lower + np.random.rand(len(joints)) * (upper - lower)
            for i, j in enumerate(joints):
                p.resetJointState(robot_id, j, seed[i])

        q = p.calculateInverseKinematics(robot_id, ee_link, pos, orn)
        q = list(q[:len(joints)])

        if not collision_fn(q):
            print(f"  IK found collision-free config on attempt {attempt+1}")
            return q

    print("  WARNING: No collision-free IK solution found after", max_attempts, "attempts")
    return None
def main():
    connect(use_gui=True)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.4, cameraYaw=90,
        cameraPitch=-42.6, cameraTargetPosition=[0.0, 0.0, 0.0]
    )

    p.setGravity(0, 0, -9.8)

    start_pos = [0.4, -0.3, 0.85]
    start_orn = p.getQuaternionFromEuler([3.14, 0, 0])
    goal_pos  = [0.4, 0.2, 0.85]
    goal_orn  = p.getQuaternionFromEuler([3.14, 0, 0])

    table_id, obstacles = setup_environment()

    panda = PandaRobot([0, 0, 0.63])
    panda.load()

    for j in range(p.getNumJoints(panda.get_robot_id())):
        p.setCollisionFilterPair(
            panda.get_robot_id(), table_id, j, -1, enableCollision=0
        )


    panda.print_info()

    # start = [-0.687,0.212,0.187,-1.273,0.656,1.024,0.219]
    # start = [0.9994, 0.6559, -1.0307, -1.0417, -0.0625, 1.6206, 0.0]
    
    # set_joint_positions(panda.get_robot_id(), panda.get_joints(), start)
    # goal = [0,-0.785,0,-2.356,0,1.871,0.785]
    # goal = [0.968, 1.003, -1.687, -0.513, 0.0, 0.942, 0.0]
    # goal = [0.5622, -0.5594, -1.1244, -1.918, -0.7496, 1.6206, 0.0]

    # --- Solve IK with collision awareness ---
    collision_fn = get_collision_fn(
        panda.get_robot_id(),
        panda.get_joints(),
        obstacles=obstacles,
        self_collisions=False,
        distance_threshold=0.05
    )
    print("Solving start IK...")
    start = solve_ik_collision_free(
        panda.get_robot_id(), panda.get_joints(),
        start_pos, start_orn, collision_fn
    )

    print("Solving goal IK...")
    goal = solve_ik_collision_free(
        panda.get_robot_id(), panda.get_joints(),
        goal_pos, goal_orn, collision_fn
    )


    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Start in collision: {collision_fn(start)}")
    print(f"Goal in collision: {collision_fn(goal)}")

    if start is None or goal is None:
        print("Could not find valid IK. Aborting.")
        return

    # Set robot to start position FIRST
    set_joint_positions(panda.get_robot_id(), panda.get_joints(), start)
    time.sleep(0.1)

    # Visualize EE positions
    start_ee = p.getLinkState(panda.get_robot_id(), 11)[0]
    draw_sphere(start_pos, [0, 1, 0])  # green = start

    # Temporarily set goal to read its EE, then restore start
    set_joint_positions(panda.get_robot_id(), panda.get_joints(), goal)
    goal_ee = p.getLinkState(panda.get_robot_id(), 11)[0]
    draw_sphere(goal_pos, [1, 0, 0])   # red = goal

    # --- CRITICAL: restore robot to START before planning ---
    set_joint_positions(panda.get_robot_id(), panda.get_joints(), start)
    p.stepSimulation()  # Add simulation step to update collision state
    time.sleep(0.1)

    print(f"Start config: {[round(v,4) for v in start]}")
    print(f"Goal  config: {[round(v,4) for v in goal]}")
    print(f"Start EE pos: {[round(v,4) for v in start_ee]}")
    print(f"Goal  EE pos: {[round(v,4) for v in goal_ee]}")

    # --- Plan ---
    path = plan_joint_motion(
        panda.get_robot_id(),
        panda.get_joints(),
        goal,
        obstacles=obstacles,
        self_collisions=False,
        distance_threshold=0.05
    )

    prev = None
    if path is not None:
        print(f"Path found with {len(path)} waypoints")
        for q in path:
            move_to_waypoint(panda.get_robot_id(), panda.get_joints(), q)
            ee = p.getLinkState(panda.get_robot_id(), 11)[0]
            if prev is not None:
                p.addUserDebugLine(prev, ee, [0, 0, 1], 2)
            prev = ee
    else:
        print("No path found.")

    try:
        while True:
            p.stepSimulation()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
if __name__ == "__main__":
    main()