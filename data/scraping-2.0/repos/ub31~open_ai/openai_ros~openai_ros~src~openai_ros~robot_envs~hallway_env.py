from openai_ros import robot_gazebo_env
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import rospy
import time


class HallwayEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all Robot environments.
    """

    def __init__(self):
        """Initializes a new Robot environment.
        """
        # Variables that we give through the constructor.

        # Internal Vars
        #rospy.logwarn("HallwayEnv init")
        self.controllers_list = []

        self.robot_name_space = ""

        reset_controls_bool = False

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv

        super(HallwayEnv, self).__init__(controllers_list=self.controllers_list,
                                         robot_name_space=self.robot_name_space,
                                         reset_controls=reset_controls_bool,
                                         start_init_physics_parameters=False,
                                         reset_world_or_sim="WORLD")
        self.gazebo.unpauseSim()
        # self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        rospy.Subscriber("/tom/odom", Odometry, self._tom_odom_callback)
        rospy.Subscriber("/jerry/odom", Odometry, self._jerry_odom_callback)

        rospy.Subscriber("/tom/scan_filtered", LaserScan, self._tom_laser_scan_callback)
        rospy.Subscriber("/jerry/scan_filtered", LaserScan, self._jerry_laser_scan_callback)

        rospy.Subscriber("/jerry/nav_kinect/depth/image_raw", Image, self._jerry_kinect_depth_callback)
        rospy.Subscriber("/jerry/nav_kinect/rgb/image_raw", Image, self._jerry_kinect_rgb_callback)


        self._tom_cmd_vel_pub = rospy.Publisher('/tom/cmd_vel', Twist, queue_size=1)
        self._jerry_cmd_vel_pub = rospy.Publisher('/jerry/cmd_vel', Twist, queue_size=1)

        self._check_publishers_connection()
        self.gazebo.pauseSim()

        rospy.logdebug("Finished HallwayEnv INIT...")


    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        #rospy.logwarn("_check_publishers_connection")
        rate = rospy.Rate(10)  # 10hz
        while (self._jerry_cmd_vel_pub.get_num_connections()==0) and (self._tom_cmd_vel_pub.get_num_connections()==0) and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")
        rospy.logdebug("All Publishers READY")

    def _tom_laser_scan_callback(self, data):
        #rospy.logwarn("_tom_laser_scan_callback")
        self.tom_laser_scan = data
    def get_tom_laser_scan(self):
        #rospy.logwarn("get_tom_laser_scan")
        return self.tom_laser_scan

    def _jerry_laser_scan_callback(self, data):
        #rospy.logwarn("_jerry_laser_scan_callback")
        self.jerry_laser_scan = data
    def get_jerry_laser_scan(self):
        #rospy.logwarn("get_jerry_laser_scan")
        return self.jerry_laser_scan

    def _jerry_kinect_depth_callback(self, data):
        #rospy.logwarn("_jerry_kinect_depth_callback")
        self.jerry_kinect_data = data
    def _get_jerry_kinect_depth(self):
        #rospy.logwarn("_get_jerry_kinect_depth")
        return self.jerry_kinect_data

    def _jerry_kinect_rgb_callback(self, data):
        #rospy.logwarn("_jerry_kinect_rgb_callback")
        self.jerry_kinect_rgb = data
    def _get_jerry_kinect_rgb(self):
        #rospy.logwarn("_get_jerry_kinect_rgb")
        return self.jerry_kinect_rgb

    def _tom_odom_callback(self,data):
        #rospy.logwarn("_tom_odom_callback")
        self.tom_odom = data
        
    def get_tom_odom(self):
        #rospy.logwarn("get_tom_odom")
        return self.tom_odom

    def _jerry_odom_callback(self, data):
        #rospy.logwarn("_jerry_odom_callback")
        self.jerry_odom = data
    def get_jerry_odom(self):
        #rospy.logwarn("get_jerry_odom")
        return self.jerry_odom

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        #rospy.logwarn("_check_all_systems_ready")
        self._check_all_sensors_ready()
        return True

    def _check_all_sensors_ready(self):
        #rospy.logwarn("_check_all_sensors_ready")
        rospy.logdebug("START ALL SENSORS READY")
        self._check_odom_ready()
        # We dont need to check for the moment, takes too long
        #self._check_camera_depth_image_raw_ready()
        #self._check_camera_depth_points_ready()
        #self._check_camera_rgb_image_raw_ready()
        self._check_laser_scan_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_odom_ready(self):
        #rospy.logwarn("_check_odom_ready")
        self.tom_odom = None
        self.jerry_odom = None
        rospy.logdebug("Waiting for /odom to be READY...")
        while (self.tom_odom is None) and (self.jerry_odom is None) and not rospy.is_shutdown():
            try:
                self.tom_odom = rospy.wait_for_message("tom/odom", Odometry, timeout=10.0)
                self.jerry_odom = rospy.wait_for_message("jerry/odom", Odometry, timeout=10.0)
                rospy.logdebug("Current /odom READY=>")
            except:
                rospy.logerr("Tom and Jerry /odom not ready yet, retrying for getting odom")

        return [self.tom_odom,self.jerry_odom]

    def _check_laser_scan_ready(self):
        #rospy.logwarn("_check_laser_scan_ready")
        self.tom_laser_scan = None
        self.jerry_laser_scan = None
        rospy.logdebug("Waiting for /scan to be READY...")
        while (self.tom_laser_scan is None) and (self.jerry_laser_scan is None) and not rospy.is_shutdown():
            try:
                self.tom_laser_scan = rospy.wait_for_message("/tom/scan", LaserScan, timeout=5.0)
                self.jerry_laser_scan = rospy.wait_for_message("/jerry/scan", LaserScan, timeout=5.0)
                rospy.logdebug("Current /scan READY=>")
            except:
                rospy.logerr("Current /kobuki/laser/scan not ready yet, retrying for getting laser_scan")
        return [self.tom_laser_scan,self.jerry_laser_scan]

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
        #rospy.logwarn(" Jerry move_base")
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("Segbot Jerry Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._jerry_cmd_vel_pub.publish(cmd_vel_value)
        time.sleep(0.2)
    
    def tom_move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10, min_laser_distance=-1):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        #rospy.logwarn("Tom move_base")
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("Segbot Tom Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._tom_cmd_vel_pub.publish(cmd_vel_value)
        time.sleep(0.2)

