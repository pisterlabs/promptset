#!/usr/bin/env python

import rospy
import time
import sys
import numpy as np

import moveit_commander
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import GetWorldProperties, GetModelState
from actionlib_msgs.msg import GoalStatusArray

from openai_ros import robot_gazebo_env
from openai_ros.openai_ros_common import ROSLauncher
from multi_geo_robot_desc.common import armrobot_creator, armrobot_controller_creator

class MultiGeoRobotEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all MultiGeoRobot environments.
    """
    

    def __init__(self, ros_ws_abspath):
        
        rospy.loginfo("Start MultiGeoRobotEnv INIT...")
        
        ########## Robot Configuration #########
        ########################################
        robotType = rospy.get_param("/multi_geo_robot/robot_type", "2D")
        arm_num = rospy.get_param("/multi_geo_robot/arm_num", 2)
        arm_lengths = rospy.get_param("/multi_geo_robot/arm_length", [0.2, 0.2])
        arm_mass = rospy.get_param("/multi_geo_robot/arm_mass", [9, 5])
        arm_radius = 0.05
        adapter_mass = rospy.get_param("/multi_geo_robot/adapter_mass", 5)
        dummy = rospy.get_param("/multi_geo_robot/eef_dummy", True)

        # Joint Position Default (6 Joints)
        # if 2D, the joint only 
        # armJoints = [[1,0,0],[0,1,0]]
        armJoints = rospy.get_param("/multi_geo_robot/arm_joint_seq", [[1,0,0],[0,1,0]])
        tooltipJoints = 1
        gripperOn = True

        ########################################
        # Generate The Robot
        self.links, self.joints = armrobot_creator(robotType,
                                                arm_num,
                                                arm_lengths,
                                                arm_mass,
                                                arm_radius,
                                                adapter_mass,
                                                armJoints,
                                                tooltipJoints,
                                                gripperOn,
                                                dummy)

        armrobot_controller_creator(self.joints,gripperOn)
        
        # We launch the ROSlaunch that spawns the robot into the world
        self.launch = ROSLauncher(rospackage_name="multi_geo_robot_desc",
                    launch_file_name="multi_geo_robot.launch",
                    ros_ws_abspath=ros_ws_abspath)
        
        # Internal Vars
        self.controllers_list = []
        
        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(MultiGeoRobotEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")

        self._gripper = rospy.Publisher('/gripper_controller/command', String, 
                                            queue_size=1)

        rospy.logdebug("MultiGeoRobotEnv unpausing")
        self.gazebo.unpauseSim()
                
        self._check_all_systems_ready()
        
        rospy.Subscriber("/joint_states", JointState, self._joints_state_callback)
        rospy.Subscriber("/move_group/status", GoalStatusArray, self._plan_status_feedback)
        rospy.Subscriber('/right_inner_finger_pad_bumper/', ContactsState, self._right_contact_state_callback)
        rospy.Subscriber('/left_inner_finger_pad_bumper/', ContactsState, self._left_contact_state_callback)
        self.last_contact_r = 0
        self.last_contact_l = 0
        
        self._move = MoveIt(self.joints)
        self._object = Obj_Pos()
        
        rospy.logdebug("MultiGeoRobotEnv pausing")
        rospy.logdebug("Finished MultiGeoRobotEnv INIT...")
        
    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def close(self):
        self.launch.launch.shutdown()
        super().close()
        
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        rospy.logdebug("MultiGeoRobotEnv check_all_systems_ready...")
        self._check_all_publishers_ready()
        self._check_all_sensors_ready()
        rospy.logdebug("END MultiGeoRobotEnv _check_all_systems_ready...")
        return True

    def _check_all_publishers_ready(self):
        rospy.logdebug("START ALL PUBLISHER")
        self._check_gripper_pub_ready()
        rospy.logdebug("All Publishers READY")
    
    def _check_gripper_pub_ready(self):
        rate = rospy.Rate(10)  # 10hz
        while (self._gripper.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug(
                "No susbribers to _gripper yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_gripper Publisher Connected")
        
    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_joint_states_ready()
        rospy.logdebug("ALL SENSORS READY")
        
    def _JointState_numpy(self, data):
        data.name = list(data.name)
        data.position = list(data.position)
        finger_index = data.name.index('finger_joint')
        manipulator_index = [data.name.index(joint_name) for joint_name in self.joints]
                                
        manipulator_joints_position = [data.position[i] for i in manipulator_index]
                
        gripper_joints_position = data.position[finger_index]
        return manipulator_joints_position, gripper_joints_position
        
    def _check_joint_states_ready(self): #check published joint names and rearrange them for the move joint_goal
        self.joint_states = None
        self.gripper_states = None
        rospy.logdebug("Waiting for /joint_states to be READY...")
        while self.joint_states is None and not rospy.is_shutdown():
            try:
                self.joint_states, self.gripper_states = self._JointState_numpy(rospy.wait_for_message("/joint_states", JointState, timeout=2.0))
                rospy.logdebug("Current /joint_states READY=>")

            except:
                rospy.logerr("Current /joint_states not ready yet, retrying for getting joint_states")
        return self.joint_states

    def _no_motion_plan(self, data):
        if data.status_list:
            if data.status_list[-1].text in "No motion plan found. No execution attempted.":
                return True
        return False
        
    def _joints_state_callback(self, data):
        self.joint_states, self.gripper_states = self._JointState_numpy(data)
        return 

    def _plan_status_feedback(self, data):
        self.no_motion_plan = self._no_motion_plan(data)
        return
        
    def _right_contact_state_callback(self, data):
        for state in data.states:
            contact_col_name = (state.collision1_name if state.collision1_name !=
                "robot::right_inner_finger::right_inner_finger_pad_collision_collision_1"
                else state.collision2_name)
            model_name = contact_col_name.split("::")[0]
            if model_name == "random_target":
                self.last_contact_r = time.time()
        return
    
    def _left_contact_state_callback(self, data):
        for state in data.states:
            contact_col_name = (state.collision1_name if state.collision1_name !=
                "robot::left_inner_finger::left_inner_finger_pad_collision_collision_1"
                else state.collision2_name)
            model_name = contact_col_name.split("::")[0]
            if model_name == "random_target":
                self.last_contact_l = time.time()
        return
        
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """
        Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time 
        we reset at the start of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """
        Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """
        Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """
        Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def open_hand(self):
        """
        When called it opens robots hand
        """
        self._gripper.publish("open")
       
    def close_hand(self):
        """
        When called it closes robots hand
        """
        self._gripper.publish("close")
        
    def get_joint_states(self): 
        self.gazebo.unpauseSim()
        joint = self.joint_states
        return joint 
        
    def get_gripper_states(self):
        self.gazebo.unpauseSim()
        joint = self.gripper_states
        return joint 
    
    def get_ee_pose(self): 
        """
        Returns the endeffector pose
        We make sure the simulation is unpaused because this calss is a service call.
        """
        rospy.logdebug("START getting endeffector pose")
        self.gazebo.unpauseSim()
        gripper_pose = self._move.ee_pose()
        rospy.logdebug("END getting endeffector pose")
        return gripper_pose   
        
    def get_ee_rpy(self): 
        rospy.logdebug("START getting endeffector rpy")
        self.gazebo.unpauseSim()
        gripper_rpy = self._move.ee_rpy()
        rospy.logdebug("END getting endeffector rpy")
        return gripper_rpy
        
    def move_ee(self, desired_pose): 
        """
        Moves the endeffector to the pose given,
        relative pose to world frame
        Currently it is only accepting position
        """
        pose_goal = Pose()
        pose_goal.position.x = desired_pose[0]
        pose_goal.position.y = desired_pose[1]
        pose_goal.position.z = desired_pose[2]
        pose_goal.orientation.x = desired_pose[3]
        pose_goal.orientation.y = desired_pose[4]
        pose_goal.orientation.z = desired_pose[5]
        pose_goal.orientation.w = desired_pose[6]
        
        
        result = self._move.ee_traj(pose_goal)
        return result
           
    def move_joints(self, joint_goal): 
        """
        Moves the joints of the robot directly
        """
        
        result = self._move.joint_traj(joint_goal)
        return result
    
    def check_pose(self, pose):
        pos = Pose()
        pos.position.x = pose[0]
        pos.position.y = pose[1]
        pos.position.z = pose[2]
        pos.orientation.x = pose[3]
        pos.orientation.y = pose[4]
        pos.orientation.z = pose[5]
        pos.orientation.w = pose[6]
        
        return self._move.check_pos(pos)[0]
    
    def get_observation(self):
        obs = self._get_obs()
        return obs
    
    def get_random_target_pos(self):
        return self._object.get_states()
        
class Obj_Pos(object):
    """
    This object maintains the pose and rotation of the object in a simulation 
    through Gazebo Service
    """

    def __init__(self):
        world_specs = rospy.ServiceProxy(
            '/gazebo/get_world_properties', GetWorldProperties)()
        self.time = 0
        self.model_names = world_specs.model_names
        self.get_model_state = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)

    def get_states(self):
        """
        Returns the ndarray of pose&rotation of the cube
        """
        for model_name in self.model_names:
            if model_name == "random_target":
                data = self.get_model_state(
                    model_name, "world")  # gazebo service client
                return np.array([
                    data.pose.position.x,
                    data.pose.position.y,
                    data.pose.position.z,
                    data.pose.orientation.x,
                    data.pose.orientation.y,
                    data.pose.orientation.z,
                    data.pose.orientation.w
                ])

class MoveIt(object):
    def __init__(self, p_joints):
        rospy.logdebug("===== MoveIt")
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("arm")
        self.move_group.set_num_planning_attempts(1)
        self.move_group.allow_replanning(False)
        self.eef_link = self.move_group.get_end_effector_link()
        self.joints = p_joints
        #self.grasping_links = self.robot.get_link_names(group='endeffector')
        #self.grasping_group = moveit_commander.MoveGroupCommander('endeffector')
        self.joint_index()
        
        rospy.logdebug("============")

    def ee_traj(self, goal):
        self.move_group.clear_pose_targets()
        self.move_group.set_pose_target(goal)
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        return plan

    def joint_traj(self, positions_array):
        self.move_group.clear_pose_targets()
        positions_array = [positions_array[i] for i in self.joint_name_index]
        result = self.move_group.go(positions_array, wait=True)
        self.move_group.stop()
        return result
        
    def robot_joints(self):
        return self.move_group.get_current_joint_values()
    
    def check_pos(self, pose):
        return self.move_group.plan(pose)
    
    def joint_index(self):
        active_joints = self.move_group.get_active_joints()
        self.joint_name_index = [active_joints.index(joint_name) for joint_name in self.joints]
        return True
        
    def ee_pose(self):
        gripper_pose = self.move_group.get_current_pose().pose
        return gripper_pose

    def ee_rpy(self, request):
        gripper_rpy = self.move_group.get_current_rpy()
        return gripper_rpy