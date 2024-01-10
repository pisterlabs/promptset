import numpy
import rospy
from openai_ros import my_robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from openai_ros.openai_ros_common import ROSLauncher


class TurtleBot3TwoRobotsEnv(my_robot_gazebo_env.MyRobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, ros_ws_abspath, ros_launch_file_package="coop_mapping", ros_launch_file_name="spawn_2_robots.launch"):
        """
        This is my own custom environment for two TB3 robots, based on the original OpenAI ROS environments.

        In this environment, the observation will be the union of the observations between both robots. 
        
        If one of them crashes, the episode ends.

        The namespace for each robot is "tb3_0" and "tb3_1", which come from the "spawn_2_robots.launch" from "coop_mapping".
        The namespaces are sometimes hardcoded in the functions, so take care when changing things.
        """
        rospy.loginfo("Start TurtleBot3TwoRobotsEnv INIT...")
    
        # Init namespace
        ROBOT_1_NAMESPACE = '/tb3_0'
        ROBOT_2_NAMESPACE = '/tb3_1'
        self.robot_namespaces = [ROBOT_1_NAMESPACE, ROBOT_2_NAMESPACE]

        # Init dictonaries for sensors
        self.odom = {}
        self.laser_scan = {}

        # Init dictionaries for publishers
        self._cmd_vel_pub = {}

        # Variables that we give through the constructor.
        # None in this case

        # We launch the ROSlaunch that spawns the two robots into the world
        ROSLauncher(rospackage_name=ros_launch_file_package,
                    launch_file_name=ros_launch_file_name,
                    ros_ws_abspath=ros_ws_abspath)

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace (The environment itself is not in a namespace. The robots are.)
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(TurtleBot3TwoRobotsEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False)



        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers.
        # This is hardcoded because I can't mess with callback arguments
        rospy.Subscriber("/tb3_0/odom", Odometry, self._odom_callback_tb3_0)
        rospy.Subscriber("/tb3_0/scan", LaserScan, self._laser_scan_callback_tb3_0)

        rospy.Subscriber("/tb3_1/odom", Odometry, self._odom_callback_tb3_1)
        rospy.Subscriber("/tb3_1/scan", LaserScan, self._laser_scan_callback_tb3_1)

        for ns in self.robot_namespaces:
            self._cmd_vel_pub[ns] = rospy.Publisher(ns + '/cmd_vel', Twist, queue_size=1)
            self._check_publishers_connection(ns)

        self.gazebo.pauseSim()

        # Variable for crash
        self._crashed = False

        rospy.loginfo("Finished TurtleBot3TwoRobotsEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.loginfo("START ALL SENSORS READY")
        for ns in self.robot_namespaces:
            self._check_odom_ready(ns)
            self._check_laser_scan_ready(ns)
        rospy.loginfo("ALL SENSORS READY")


    def _check_odom_ready(self, namespace):
        self.odom[namespace] = None
        rospy.loginfo("Waiting for {}/odom to be READY...".format(namespace))
        while self.odom[namespace] is None and not rospy.is_shutdown():
            try:
                self.odom[namespace] = rospy.wait_for_message(namespace + "/odom", Odometry, timeout=5.0)
                rospy.loginfo("Current {}/odom READY=>".format(namespace))

            except:
                rospy.logerr("Current {}/odom not ready yet, retrying for getting odom".format(namespace))

        return self.odom[namespace]

    def _check_laser_scan_ready(self, namespace):
        self.laser_scan[namespace] = None
        rospy.loginfo("Waiting for {}/scan to be READY...".format(namespace))
        while self.laser_scan[namespace] is None and not rospy.is_shutdown():
            try:
                self.laser_scan[namespace] = rospy.wait_for_message(namespace + "/scan", LaserScan, timeout=1.0)
                rospy.loginfo("Current {}/scan READY=>".format(namespace))

            except:
                rospy.logerr("Current {}/scan not ready yet, retrying for getting laser_scan".format(namespace))
        return self.laser_scan[namespace]
        
    # TB3_0
    def _odom_callback_tb3_0(self, data):
        self.odom['/tb3_0'] = data

    def _laser_scan_callback_tb3_0(self, data):
        self.laser_scan["/tb3_0"] = data

    # TB3_1
    def _odom_callback_tb3_1(self, data):
        self.odom['/tb3_1'] = data

    def _laser_scan_callback_tb3_1(self, data):
        self.laser_scan['/tb3_1'] = data

        
    def _check_publishers_connection(self, namespace):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub[namespace].get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.loginfo("No susbribers to _cmd_vel_pub of {} yet so we wait and try again".format(namespace))
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.loginfo("_cmd_vel_pub Publisher of {} Connected".format(namespace))

        rospy.loginfo("All Publishers READY")
    
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
    def move_base(self, linear_speed, angular_speed, namespace, epsilon=0.05, update_rate=10):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return: 
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.loginfo("TurtleBot3 Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection(namespace)
        self._cmd_vel_pub[namespace].publish(cmd_vel_value)
        self.wait_until_twist_achieved(cmd_vel_value,
                                        epsilon,
                                        update_rate,
                                        namespace)

        # After moving, stop, so the other robots can move
        stop_twist = Twist()
        self._cmd_vel_pub[namespace].publish(stop_twist)
        self.wait_until_twist_achieved(stop_twist,
                                        epsilon,
                                        update_rate,
                                        namespace)

    
    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate, namespace):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        rospy.loginfo("START wait_until_twist_achieved...")
        
        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0
        epsilon = 0.05
        
        rospy.loginfo("Desired Twist Cmd>>" + str(cmd_vel_value))
        rospy.loginfo("epsilon>>" + str(epsilon))
        
        linear_speed = cmd_vel_value.linear.x
        angular_speed = cmd_vel_value.angular.z
        
        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon
        angular_speed_plus = angular_speed + epsilon
        angular_speed_minus = angular_speed - epsilon
        
        while not rospy.is_shutdown():
            current_odometry = self._check_odom_ready(namespace)

            # In turtlebot3 the odometry angular readings are inverted, so we have to invert the sign.
            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = current_odometry.twist.twist.angular.z
            
            rospy.loginfo("Linear VEL of {}=".format(namespace) + str(odom_linear_vel) + ", ?RANGE=[" + str(linear_speed_minus) + ","+str(linear_speed_plus)+"]")
            rospy.loginfo("Angular VEL of {}=".format(namespace) + str(odom_angular_vel) + ", ?RANGE=[" + str(angular_speed_minus) + ","+str(angular_speed_plus)+"]")
            
            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
            angular_vel_are_close = (odom_angular_vel <= angular_speed_plus) and (odom_angular_vel > angular_speed_minus)
            
            if linear_vel_are_close and angular_vel_are_close:
                rospy.loginfo("{} reached Velocity!".format(namespace))
                end_wait_time = rospy.get_rostime().to_sec()
                break
            
            if self.check_if_crashed(namespace):
                rospy.logerr("{} has crashed while trying to achieve Twist.".format(namespace))
                self._episode_done = True
                self._crashed = True
                break

            rospy.loginfo("{} is not there yet, keep waiting...".format(namespace))
            rate.sleep()
            
        delta_time = end_wait_time - start_wait_time
        rospy.loginfo("[Wait Time=" + str(delta_time)+"]")
        
        rospy.loginfo("END wait_until_twist_achieved...")
        
        return delta_time