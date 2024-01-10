#! /usr/bin/env python

import numpy
import rospy
from openai_ros.openai_ros.src.openai_ros import robot_gazebo_env_goal
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import sys
from moveo_training.moveo_commander.moveo_commander import MoveoCommander
import geometry_msgs.msg

class MoveoEnv(robot_gazebo_env_goal.RobotGazeboEnv):
    """Superclass for all Moveo environments.
    """

    def __init__(self):
        # print ("Entered Moveo Env")
        """Initializes a new Moveo environment.

        Args:
            
        """


        """
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that the stream of data doesn't flow. This is for simulations
        that are paused for whatever reason
        2) If the simulation was running already for some reason, we need to reset the controllers.
        This has to do with the fact that some plugins with tf don't understand the reset of the simulation
        and need to be reset to work properly.
        """

        # We Start all the ROS related Subscribers and publishers
        
        JOINT_STATES_SUBSCRIBER = '/joint_states'
        
        self.joint_states_sub = rospy.Subscriber(JOINT_STATES_SUBSCRIBER, JointState, self.joints_callback)
        self.joints = JointState()
        
        # We start the moveo commander object
        self.moveo_commander_obj = MoveoCommander()
        # print(type(self.moveo_commander_obj))
        # Variables that we give through the constructor.

        self.controllers_list = []

        self.robot_name_space = ""
        
        # We launch the init function of the Parent Class robot_gazebo_env_goal.RobotGazeboEnv
        # print ("launch the init function of the Parent Class robot_gazebo_env_goal.RobotGazeboEnv.....")
        super().__init__(controllers_list=self.controllers_list,
                                                robot_name_space=self.robot_name_space,
                                                reset_controls=False)
        # print ("launch the init function of the Parent Class robot_gazebo_env_goal.RobotGazeboEnv.....END")
        
        # print ("Entered Moveo Env END")



    # RobotGazeboEnv virtual methods
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    # MoveoEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        self._check_joint_states_ready()
        
        # rospy.logdebug("ALL SENSORS READY")

    def _check_joint_states_ready(self):
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message("/joint_states", JointState, timeout=1.0)
                # rospy.logdebug("Current /joint_states READY=>" + str(self.joints))

            except:
                rospy.logerr("Current /joint_states not ready yet, retrying for getting joint_states")
        return self.joints
    
    def joints_callback(self, data):
        self.joints = data

    def get_joints(self):
        return self.joints

    def set_trajectory_ee(self, action):
        """
        Sets the enf effector position and orientation
        """
        
        ee_pose = geometry_msgs.msg.Pose()
        ee_pose.position.x = action[0]
        ee_pose.position.y = action[1]
        ee_pose.position.z = action[2]
        ee_pose.orientation.x = 0.0
        ee_pose.orientation.y = 0.0
        ee_pose.orientation.z = 0.0
        ee_pose.orientation.w = 1.0
        self.moveo_commander_obj.move_ee_to_pose(ee_pose)

        return True
        
    def set_trajectory_joints(self, initial_qpos):
        """
        Helper function.
        Wraps an action vector of joint angles into a JointTrajectory message.
        The velocities, accelerations, and effort do not control the arm motion
        """

        position = [None] * 5
        # print(initial_qpos["Joint_1"])
        # print(initial_qpos["Joint_2"])
        # print(initial_qpos["Joint_3"])
        # print(initial_qpos["Joint_4"])
        # print(initial_qpos["Joint_5"])
        position[0] = initial_qpos["Joint_1"]
        position[1] = initial_qpos["Joint_2"]
        position[2] = initial_qpos["Joint_3"]
        position[3] = initial_qpos["Joint_4"]
        position[4] = initial_qpos["Joint_5"]
      
        #position = [ 0.0, 0.0, 0.0, 0.0, 0.0]
        try:
            self.moveo_commander_obj.move_joints_traj(position)
            result = True
        except Exception as ex:
            print(ex)
            result = False

        return result
        
    def get_ee_pose(self):

        gripper_pose = self.moveo_commander_obj.get_ee_pose()
        
        return gripper_pose
        
    def get_ee_rpy(self):
        
        gripper_rpy = self.moveo_commander_obj.get_ee_rpy()
        
        return gripper_rpy
    
   
    # ParticularEnv methods
    # ----------------------------

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()