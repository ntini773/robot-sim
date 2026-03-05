import pybullet as p
import pybullet_data
import time

def create_box(position, size=[0.1,0.1,0.1], color=[1,0,0,1]):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )
    return body

def setup_environment():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useMaximalCoordinates=True)
    table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
    box1 = create_box(position=[0.55, 0, 1.2], size=[0.25, 0.08, 0.15], color=[1, 0, 0, 1])
    obstacles = [plane_id, table_id, box1]
    return table_id, obstacles

def print_camera_info(cam):
    """Print camera params in a ready-to-hardcode format."""
    print(f"Camera | distance={round(cam[10], 3)}  yaw={round(cam[8], 2)}  pitch={round(cam[9], 2)}  target={[round(v, 4) for v in cam[11]]}")
    print(f"  → p.resetDebugVisualizerCamera(cameraDistance={round(cam[10], 3)}, cameraYaw={round(cam[8], 2)}, cameraPitch={round(cam[9], 2)}, cameraTargetPosition={[round(v, 4) for v in cam[11]]})")

# Connect to PyBullet GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

table_id, obstacles = setup_environment()
# robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.63], useFixedBase=True)
robot = p.loadURDF("./lite-6-updated-urdf/lite_6_new.urdf", [0, 0, 0.7], useFixedBase=True)
num_joints = p.getNumJoints(robot)

sliders = {}
joint_ids = []

for i in range(num_joints):
    info = p.getJointInfo(robot, i)
    joint_name = info[1].decode()
    joint_type = info[2]
    lower = info[8]
    upper = info[9]

    if joint_type == p.JOINT_FIXED:
        continue

    if lower > upper:
        lower, upper = -3.14, 3.14

    slider = p.addUserDebugParameter(joint_name, lower, upper, 0)
    sliders[i] = slider
    joint_ids.append(i)
    print("Joint:", i, joint_name)

CHANGE_THRESHOLD = 0.001
CAMERA_THRESHOLD = 0.01

prev_config = [0.0] * len(joint_ids)
prev_cam = p.getDebugVisualizerCamera()

# Main control loop
while True:
    current_config = []

    for j in joint_ids:
        target = p.readUserDebugParameter(sliders[j])
        current_config.append(target)
        p.setJointMotorControl2(
            robot, j,
            p.POSITION_CONTROL,
            targetPosition=target,
            force=200
        )

    # Print joint config on change
    if any(abs(current_config[k] - prev_config[k]) > CHANGE_THRESHOLD for k in range(len(joint_ids))):
        formatted = [round(v, 4) for v in current_config]
        print(f"Joint config: {formatted}")
        prev_config = current_config[:]

    # Print camera info on change
    cam = p.getDebugVisualizerCamera()
    if (abs(cam[10] - prev_cam[10]) > CAMERA_THRESHOLD or   # distance
        abs(cam[8]  - prev_cam[8])  > CAMERA_THRESHOLD or   # yaw
        abs(cam[9]  - prev_cam[9])  > CAMERA_THRESHOLD or   # pitch
        abs(cam[11][0] - prev_cam[11][0]) > CAMERA_THRESHOLD): # target
        print_camera_info(cam)
        prev_cam = cam

    p.stepSimulation()
    time.sleep(1/240)