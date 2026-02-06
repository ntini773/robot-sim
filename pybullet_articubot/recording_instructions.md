# How to Record a Pick and Place Task

This document explains how to record simulation data for a pick and place task using the `pybullet_articubot` environment.

## Overview

The recording process involves running the simulation loop, executing robot actions (either via motion planning or direct control), and saving the state of the environment at each time step.

## Relevant Classes

### 1. `SimpleEnv` (`sim.py`)
This is the core environment class.
- **Role**: Manages the physics engine (PyBullet), loads the robot (`UR5Robotiq85`), and loads scene objects.
- **Key Parameters**:
  - `control_step` (default: 5): Determines granular simulation steps per control action.
  - `dt` (default: 1/240): The physics time step.

### 2. `PointCloudWrapper` (`wrapper.py`)
This wrapper sits on top of `SimpleEnv` to generate sensor observations.
- **Role**: Captures depth images from multiple camera views, fuses them into a 3D point cloud, and downsamples it.
- **Output**: A dictionary containing `point_cloud`, `agent_pos`, and `gripper_pcd`.

### 3. `StateIO` (`utils/state_io.py`)
This utility module handles the actual data serialization.
- **Role**: Provides functions to save the entire environment state (robot joints + object poses) or a trajectory of observations.
- **Key Functions**:
  - `save_env(env, path)`: Snapshots the exact state of physics bodies.
  - `save_trajectory(path, trajectory)`: Saves a list of observation dictionaries to a pickle file.

## Data Structure

When you record a task, you typically save a **Trajectory**. A trajectory is a list of dictionaries, where each dictionary represents one time step.

### Recorded Data Fields
1. **`point_cloud`**: `(N, 3)` NumPy array of float32.
   - Represents the 3D scene geometry as seen by the cameras.
   - Filtered to remove the floor and distant background.
2. **`agent_pos`**: `(13,)` NumPy array.
   - Contains the robot's joint angles (including gripper) and potentially base pose.
3. **`gripper_pcd`**: `(4, 3)` NumPy array.
   - Keypoints representing the current position of the gripper fingers.
4. **`action`**: (Optional) The action taken at that step.
5. **`object_states`**: (Optional, if using `save_env`)
   - exact [x,y,z] position and quaternion orientation of the target object.

## frequency

The recording frequency is determined by the **Control Frequency**, not the Physics Frequency.

- **Physics Frequency**: 240 Hz (`dt = 1/240`).
- **Control Frequency**: ~50 Hz (240 Hz / `control_step=5`).

Data is recorded every time you call `env.step(action)` or manually trigger a recording loop during motion execution.

## Example: How to Record

To record a pick and place task, you would modify your main script (`demo.py`) as follows:

```python
import pickle
from utils.state_io import save_trajectory

# 1. Initialize List
trajectory = []

# 2. In your execution loop
success, path, _, _ = motion_planning(...)

if success:
    for q in path:
        # Move robot
        env.env.robot.control(env.env.robot.arm_joint_indices, q)
        
        # Step Physics
        for _ in range(5):
            p.stepSimulation()
            
        # 3. Capture Observation
        # Note: We use the wrapper's observation method
        raw_state = env.env.robot.get_state()
        obs = env.observation(raw_state)
        
        # 4. Append to list
        trajectory.append(obs)

# 5. Save to disk
save_trajectory("pick_place_demo.pkl", trajectory)
print(f"Saved {len(trajectory)} frames.")
```
