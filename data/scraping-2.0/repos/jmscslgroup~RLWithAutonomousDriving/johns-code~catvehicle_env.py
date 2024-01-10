import gym
import rospy
import roslaunch
import time
import numpy as np
from gym import utils, spaces
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gym.utils import seeding
from gym.envs.registration import register
import copy
import math
import os
# I will assuem joint#_velocity_controller is from sensor_msgs.msg
from sensor_msgs.msg import JointState, joint1_velocity_controller, joint2_velocity_controller, front_left_steering_position_controller, front_right_steering_position_controller
from std_msgs.msg import Float64
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkState
from rosgraph_msgs.msg import Clock
from openai_ros import robot_gazebo_env


# TODO:
#   1. Import .msg associated with joint1_velocity_controller

class MyRobotEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all Robot environments.
    """

    def __init__(self):
        """Initializes a new Robot environment.
        """
        
        """
        Initializes a new CATVehicle environment.
        Turtlebot2 doesnt use controller_manager, therefore we wont reset the 
        controllers in the standard fashion. For the moment we wont reset them.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that the stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        
        The Sensors: The sensors accesible are the ones considered usefull for AI learning.
        
        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot
        * /camera/depth/image_raw: 2d Depth image of the depth sensor.
        * /camera/depth/points: Pointcloud sensor readings
        * /camera/rgb/image_raw: RGB camera
        * /kobuki/laser/scan: Laser Readings
        
        Actuators Topic List: /cmd_vel, 
        
        Args:
        """
        
        rospy.logdebug("Start CATVehicle_ENV INIT...")
        
        # These 4 publishers control the 4 wheels of the car
        self.publishers_array = []
        
        #self._bl = rospy.Publisher("/catvehicle/joint1_velocity_controller/command", Float32, self.back_left_vel)
        #self._br = rospy.Publisher("/catvehicle/joint2_velocity_controller/command", Float32, self.back_right_vel)
        #self._fl = rospy.Publisher("/catvehicle/front_left_steering_position_controller/command", Float 32, self.front_left_steering)
        #self._fr = rospy.Publisher("/catvehicle/front_right_steering_position_controller/command", Float 32, self.front_right_steering)
        
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        #self.publishers_array.append(self._br)
        #self.publishers_array.append(self._bl)
        #self.publishers_array.append(self._fr)
        #self.publishers_array.append(self._fl)
        
        rospy.Subscriber("/catvehicle/distanceEstimatorSteeringBased/dist", Float32, self.dist)
        rospy.Subscriber("/catvehicle/distanceEstimatorSteeringBased/angle", Float32, self.angle)
        #rospy.Subscriber("/catvehicle/joint_states", JointState, self.joints_callback)
        
        
        
        #self.controllers_list = ['joint1_velocity_controller', 
        #                         'joint2_velocity_controller',  
        #                         'front_left_steering_position_controller',
        #                         'front_right_steering_position_controller',  
        #                         'joint_state_controller']
        
        self.controllers_list = []

        self.robot_name_space = "catvehicle_v0"

        self.reset_controls = True
        
        # Seed the environment
        self._seed()
        self.steps_beyond_done = None
        
        
        
        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(CATVehicleEnv, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=self.reset_controls
            )

    
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        
        self._check_dist_ready()
        self._check_angle_ready()
        #self._check_joint_states_ready()
        self._check_cmd_vel_pub()
        
        return True
        
    # Check our distance sensor working
    def _check_dist_ready(self):
        self.dist = None
        rospy.logdebug("Waiting for /catvehicle/distanceEstimatorSteeringBased/dist to be READY...")
        while self.dist is None and not rospy.is_shutdown():
            try:
                self.dist = rospy.wait_for_message("/catvehicle/distanceEstimatorSteeringBased/dist", Float32, timeout=5.0)
                rospy.logdebug("Current /catvehicle/distanceEstimatorSteeringBased/dist READY=>")

            except:
                rospy.logerr("Current /catvehicle/distanceEstimatorSteeringBased/dist not ready yet, retrying for getting odom")

        return self.dist
        
    # Checks our angle sensor is working
    def _check_angle_ready(self):
        self.angle = None
        rospy.logdebug("Waiting for /catvehicle/distanceEstimatorSteeringBased/angle to be READY...")
        while self.angle is None and not rospy.is_shutdown():
            try:
                self.angle = rospy.wait_for_message("/catvehicle/distanceEstimatorSteeringBased/angle", Float32, timeout=5.0)
                rospy.logdebug("Current /catvehicle/distanceEstimatorSteeringBased/angle READY=>")

            except:
                rospy.logerr("Current /catvehicle/distanceEstimatorSteeringBased/angle not ready yet, retrying for getting odom")

        return self.angle
        
    # Check joint states (state of car)
    def _check_joint_states_ready():
        self.base_position = None
        while self.base_position is None and not rospy.is_shutdown():
            try:
                self.base_position = rospy.wait_for_message("/catvehicle/joint_states", JointState, timeout=1.0)
                rospy.logdebug("Current catvehicle_v0/joint_states READY=>"+str(self.base_position))
                if init:
                    # We Check all the sensors are in their initial values
                    positions_ok = all(abs(i) <= 1.0e-02 for i in self.base_position.position)
                    velocity_ok = all(abs(i) <= 1.0e-02 for i in self.base_position.velocity)
                    efforts_ok = all(abs(i) <= 1.0e-01 for i in self.base_position.effort)
                    base_data_ok = positions_ok and velocity_ok and efforts_ok
                    rospy.logdebug("Checking Init Values Ok=>" + str(base_data_ok))
            except:
                rospy.logerr("Current catvehicle_v0/joint_states not ready yet, retrying for getting joint_states")
                
    def _check_cmd_vel_pub():
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")
                
     # Check that all publishers are working
     def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        """
        rate = rospy.Rate(10)  # 10hz; HOW DOES THIS WORK FOR CATVEHICLE/
        while (self._br.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to _br yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_br Publisher Connected")

        while (self._bl.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to _bl yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_bl Publisher Connected")
        
        while (self._fr.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to _fr yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_fr Publisher Connected")

        while (self._fl.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to _fl yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_fl Publisher Connected")
        """
        
        while (self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown()):
            rospy.logdebug("No susbribers to self._cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("self._cmd_vel_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")
        
    
    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
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
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    
    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10, min_laser_distance=-1):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return: 
        """
        cmd_vel_value = Twist() # Describes linear motion and angular motion of robot
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("CATVehicle Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        time.sleep(0.2)
        #time.sleep(0.02)
        """
        self.wait_until_twist_achieved(cmd_vel_value,
                                        epsilon,
                                        update_rate,
                                        min_laser_distance)
        """
        
    def has_crashed(self, min_distance):
        """
        It states based on the laser scan if the robot has crashed or not.
        Crashed means that the minimum laser reading is lower than the
        min_laser_distance value given.
        If min_laser_distance == -1, it returns always false, because its the way
        to deactivate this check.
        """
        robot_has_crashed = False
        
        if min_distance != -1:
            dist = self.get_dist()
            if item == float ('Inf') or numpy.isinf(item):
                pass
            elif numpy.isnan(item):
               pass
            else:
                # Has a Non Infinite or Nan Value
                if (dist < min_laser_distance):
                    rospy.logerr("CATVehicle HAS CRASHED >>> item=" + str(dist)+"< "+str(min_laser_distance))
                        robot_has_crashed = True
                        break
        return robot_has_crashed
        
    def get_dist():
        return self.dist
        
    def get_angle():
        return self.angle
        
    
    
    
