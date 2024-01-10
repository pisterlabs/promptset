#! /usr/bin/env python

import numpy
import rospy
import geometry_msgs.msg
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from aubo_moveit_config.aubo_commander import AuboCommander
from openai_ros import robot_gazebo_env

class AuboEnv(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self):
        """
        Initializes a new aubo environment.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        
        The Sensors: The sensors accesible are the ones considered usefull for AI learning.
        
        Sensor Topic List:
        * /aubo_i5/camera/image
        
        Actuators Topic List: 
        * /aubo_i5/arm_controller/follow_joint_trajectory
        * /aubo_i5/gripper_controller/gripper_cmd

        
        Args:
        """
        rospy.logdebug("Start AuboEnv INIT...")

        JOINT_STATES_SUBSCRIBER = '/joint_states'
        GIPPER_IMAGE_SUBSCRIBER = '/camera/image_raw'
        self.joint_states_sub = rospy.Subscriber(JOINT_STATES_SUBSCRIBER, JointState, self.joints_callback)
        self.joints = JointState()

        self.grippper_camera_image_raw = rospy.Subscriber(GIPPER_IMAGE_SUBSCRIBER, Image, self.gripper_camera_callback)
        self.grippper_camera_image_raw = Image()

        self.controllers_list = []

        self.aubo_commander = AuboCommander()

        self.setup_planning_scene()
        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(AuboEnv, self).__init__(controllers_list=self.controllers_list,
	                                    robot_name_space=self.robot_name_space,
	                                    reset_controls=False,
	                                    start_init_physics_parameters=False,
	                                    reset_world_or_sim="WORLD")
    def setup_planning_scene(self):
        # add table mesh in scene planning, avoiding to collosion
        rospy.sleep(2)

        p = PoseStamped()
        p.header.frame_id = self.aubo_commander.robot.get_planning_frame()
        p.pose.position.x = 0.75
        p.pose.position.y = 0.
        p.pose.position.z = 0.386

        self.aubo_commander.scene.add_box("table",p,(0.91,0.91,0.77))

    def joints_callback(self, data):
        # get joint_states
        self.joints = data

    def gripper_camera_callback(self, data):
        #get camera raw
    	self.grippper_camera_image_raw = data


    def _check_all_systems_ready(self):
        """
        Checks joint_state_publisher and camera topic , publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True

    def _check_all_sensors_ready(self):
        self._check_joint_states_ready()
        self._check_gripper_camera_image_ready()
        
        rospy.logdebug("ALL SENSORS READY")

    def _check_joint_states_ready(self):
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message("/joint_states", JointState, timeout=1.0)
                rospy.logdebug("Current /joint_states READY=>" + str(self.joints))

            except:
                rospy.logerr("Current /joint_states not ready yet, retrying for getting joint_states")
        return self.joints

    def _check_gripper_camera_image_ready(self):
        self.grippper_camera_image_raw = None
        while self.grippper_camera_image_raw is None and not rospy.is_shutdown():
            try:
                self.grippper_camera_image_raw = rospy.wait_for_message("/camera/image_raw", Image , timeout=1.0)
                rospy.logdebug("Current /camera/image_raw READY" )

            except:
                rospy.logerr("Current /camera/image_raw not ready yet, retrying for getting image_raw")
        return self.grippper_camera_image_raw
    
    def get_joints(self):
    	return self.joints


    def set_trajectory_ee(self, position):
        """
        Sets the enf effector position and orientation
        """

        ee_pose = geometry_msgs.msg.Pose()
        ee_pose.position.x = position[0]
        ee_pose.position.y = position[1]
        ee_pose.position.z = position[2]
        ee_pose.orientation.x = 0.0
        ee_pose.orientation.y = 1.0
        ee_pose.orientation.z = 0.0
        ee_pose.orientation.w = 0.0
      

        return self.aubo_commander.move_ee_to_pose(ee_pose)
        
        
    def set_trajectory_joints(self, arm_joints):
        """
        Helper function.
        Wraps an action vector of joint angles into a JointTrajectory message.
        The velocities, accelerations, and effort do not control the arm motion
        """

        position = [None] * 6
        position[0] = arm_joints["shoulder_joint"]
        position[1] = arm_joints["upperArm_joint"]
        position[2] = arm_joints["foreArm_joint"]
        position[3] = arm_joints["wrist1_joint"]
        position[4] = arm_joints["wrist2_joint"]
        position[5] = arm_joints["wrist3_joint"]


        return self.aubo_commander.move_joints_traj(position)

    def get_ee_pose(self):

        gripper_pose = self.aubo_commander.get_ee_pose()
        
        return gripper_pose
        
    def get_ee_rpy(self):
        
        gripper_rpy = self.aubo_commander.get_ee_rpy()
        
        return gripper_rpy

    def set_ee(self, value):
        return self.aubo_commander.execut_ee(value)

    
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

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
