The main code is 
in `motion_planning_lite6.py`, which is a motion planning script for a robot in a simulated environment. The code sets up the simulation, defines the robot and its environment, and then uses a planner to find a trajectory from point A to point B while avoiding obstacles.

and motion_planning_lite6_data.py, which is a data collection script that runs the motion planning code multiple times to collect data on the robot's configurations, trajectories, and collisions. The collected data is saved in a structured format for later analysis. 

See config/planner_config.yaml for the configuration file that defines the parameters for the planner and the optimization process.

In motion_planning_lite6.py, I have 2 sets of obstacle setups , 
see `generate_task_obstacles_old` and `generate_task_obstacles` in the function motion_plan_data in these files.