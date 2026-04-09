import pybullet as p
import pybullet_data
from collections import namedtuple
class FrankaRobot:
    """
    Simple Franka Panda robot class for testing.
    Follows same interface pattern as Lite6Robot for compatibility with RobotOMPLPlanner.
    """
    
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 11  # Franka EEF link index
        self.arm_num_dofs = 7
        self.arm_rest_poses = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.max_velocity = 3.0
        self.id = None
    
    def load(self):
        """Load Franka Panda URDF"""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.id = p.loadURDF(
            "franka_panda/panda.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
        )
        
        self._parse_joint_info()
        self._print_debug_info()
    
    def _parse_joint_info(self):
        """Parse joint information from URDF"""
        jointInfo = namedtuple(
            "jointInfo",
            ["id", "name", "type", "lowerLimit", "upperLimit", 
             "maxForce", "maxVelocity", "controllable"],
        )
        
        self.joints = []
        self.controllable_joints = []
        
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            
            if controllable:
                self.controllable_joints.append(jointID)
            
            self.joints.append(
                jointInfo(
                    jointID, jointName, jointType,
                    jointLowerLimit, jointUpperLimit,
                    jointMaxForce, jointMaxVelocity,
                    controllable,
                )
            )
        
        # First 7 joints are arm, rest are gripper
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [
            ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)
        ]
    
    def _print_debug_info(self):
        """Print debug information"""
        print("\n" + "="*60)
        print("FRANKA ROBOT DEBUG INFO")
        print("="*60)
        jtype_names = {0: 'REVOLUTE', 1: 'PRISMATIC', 4: 'FIXED'}
        for j in self.joints:
            jtype = jtype_names.get(j.type, str(j.type))
            ctrl = "CTRL" if j.controllable else "    "
            print(f"  Joint {j.id:2d}: {j.name:<35s} ({jtype:<9s}) [{ctrl}]")
        
        print(f"\nArm controllable joints: {self.arm_controllable_joints}")
        print(f"Arm DOF: {self.arm_num_dofs}")
        print(f"EEF link index: {self.eef_id}")
        
        eef_state = p.getLinkState(self.id, self.eef_id)
        print(f"EEF position: {eef_state[0]}")
        print(f"EEF orientation (quat): {eef_state[1]}")
        print("="*60 + "\n")
    
    def get_current_ee_position(self):
        """Get current end-effector pose"""
        eef_state = p.getLinkState(self.id, self.eef_id)
        return eef_state[0], eef_state[1]
    def reset_posture(self):
        pass
