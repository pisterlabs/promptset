import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64, String
from sensor_msgs.msg import JointState, Image, LaserScan, PointCloud2, Imu
from nav_msgs.msg import Odometry, OccupancyGrid
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import Twist, PoseStamped
from visualization_msgs.msg import Marker
from openai_ros.openai_ros_common import ROSLauncher
from move_base_msgs.msg import MoveBaseActionResult
from actionlib_msgs.msg import GoalID


class TurtleBot3Env(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, ros_ws_abspath):

        self.robot_namespace = '/tb3_0'
        self.robot_namespace_no_slash = 'tb3_0'
        self.odom_topic = self.robot_namespace + "/odom"
        self.scan_topic = self.robot_namespace + "/scan"
        self.imu_topic = self.robot_namespace + "/imu"
        self.cmd_vel_topic = self.robot_namespace + "/cmd_vel"
        self.frontier_topic = self.robot_namespace+'/rl_frontiers'
        self.map_topic = self.robot_namespace+'/rl_map'
        self.pose_topic = self.robot_namespace+'/rl_pose'
        self.raw_map_topic = self.robot_namespace+'/map'
        self.reset_gmapping_topic = '/syscommand'
        self.reset_gmapping_fully_topic = '/gmapping_finished_resetting'
        self.move_base_reached_topic = self.robot_namespace + '/move_base/result'
        self.move_base_delete_goal_topic = self.robot_namespace + '/move_base/cancel'


        """
        Initializes a new TurtleBot3Env environment.
        TurtleBot3 doesnt use controller_manager, therefore we wont reset the
        controllers in the standard fashion. For the moment we wont reset them.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered usefull for AI learning.

        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot
        * /imu: Inertial Mesuring Unit that gives relative accelerations and orientations.
        * /scan: Laser Readings

        Actuators Topic List: /cmd_vel,

        Args:
        """
        rospy.loginfo("Start TurtleBot3Env INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # We launch the ROSlaunch that spawns the robot into the world
        # ROSLauncher(rospackage_name="turtlebot3_gazebo",
        #             launch_file_name="put_robot_in_world.launch",
        #             ros_ws_abspath=ros_ws_abspath)

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = ["imu"]

        # # It doesnt use namespace  -- IT DOES NOW -- at top
        # self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(TurtleBot3Env, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_namespace,
                                            reset_controls=False,
                                            start_init_physics_parameters=False)

        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        self.odom_recorder = list()
        self.odom_recorder_max = 10
        self.odom_diff_threshold = 0.001

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_callback)
        rospy.Subscriber(self.imu_topic, Imu, self._imu_callback)
        rospy.Subscriber(self.scan_topic, LaserScan, self._laser_scan_callback)
        rospy.Subscriber(self.frontier_topic, OccupancyGrid, self._frontier_callback)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self._map_callback)
        rospy.Subscriber(self.pose_topic, OccupancyGrid, self._pose_callback)
        rospy.Subscriber(self.raw_map_topic, OccupancyGrid, self._raw_map_callback)
        rospy.Subscriber(self.reset_gmapping_fully_topic, String, self._gmapping_reset_callback)
        rospy.Subscriber(self.move_base_reached_topic, MoveBaseActionResult, self._move_base_result_callback)

        self.stop_move_base_pub = rospy.Publisher('/stop_move_base', String, queue_size=10)
        self.start_move_base_pub = rospy.Publisher('/start_move_base', String, queue_size=10)

        self.reset_gmapping_pub = rospy.Publisher(self.reset_gmapping_topic, String, queue_size=10)

        self.frontier_map = OccupancyGrid()
        self.map = OccupancyGrid()
        self.pose_map = OccupancyGrid()

        self.frontier_map_received = False
        self.map_received = False
        self.pose_map_received = False

        self._cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self._check_publishers_connection()

        self.gazebo.pauseSim()
        self.gmapping_fully_reset = False

        self.move_pub = rospy.Publisher(self.robot_namespace+'/move_base_simple/goal', PoseStamped, queue_size=10)
        self.visualization_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)
        self.goal_delete_pub = rospy.Publisher(self.move_base_delete_goal_topic, GoalID, queue_size=10)

        self.move_base_threshold = 0.1
        rospy.set_param(self.robot_namespace + '/move_base/DWAPlannerROS/xy_goal_tolerance', self.move_base_threshold)

        self.record_move_base_response = False
        self.move_base_result = None

        rospy.loginfo("Finished TurtleBot3Env INIT...")


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
        rospy.logdebug("START ALL SENSORS READY")
        self._check_odom_ready()
        self._check_imu_ready()
        self._check_laser_scan_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message(self.odom_topic, Odometry, timeout=5.0)
                rospy.logdebug("Current " + self.odom_topic + " READY=>")

            except:
                rospy.logerr("Current " + self.odom_topic + " not ready yet, retrying for getting odom")

        return self.odom


    def _check_imu_ready(self):
        self.imu = None
        rospy.logdebug("Waiting for /imu to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message(self.imu_topic, Imu, timeout=5.0)
                rospy.logdebug("Current " + self.imu_topic + " READY=>")

            except:
                rospy.logerr("Current " + self.imu_topic + " not ready yet, retrying for getting imu")

        return self.imu


    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.logdebug("Waiting for /scan to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message(self.scan_topic, LaserScan, timeout=1.0)
                rospy.logdebug("Current " + self.scan_topic + " READY=>")

            except:
                rospy.logerr("Current " + self.scan_topic + " not ready yet, retrying for getting laser_scan")
        return self.laser_scan


    def _odom_callback(self, data):
        self.odom = data

        if len(self.odom_recorder) < self.odom_recorder_max:
            self.odom_recorder.append(data)
        else:
            oldest_odom = self.odom_recorder.pop(0)
            self.odom_recorder.append(data)

        # print("Odom Received")


    def _imu_callback(self, data):
        self.imu = data
        # print("IMU Received")

    def _laser_scan_callback(self, data):
        self.laser_scan = data
        # print("Laser Scan Received")

    def _frontier_callback(self, data):
        self.frontier_map = data
        self.frontier_map_received = True
        # print("Frontier Map Received")

    def _map_callback(self, data):
        self.map = data
        self.map_received = True
        # print("Full Occupancy Map Received")

    def _pose_callback(self, data):
        self.pose_map = data
        self.pose_map_received = True
        # print("Pose Map Received")

    def _raw_map_callback(self, data):
        print("Raw Map Received")

    def _gmapping_reset_callback(self, data):
        if data == "fully reset":
            self.gmapping_fully_reset = True

    def _move_base_result_callback(self, data):
        if self.record_move_base_response:
            self.move_base_result = data

    def _check_publishers_connection(self):
        """
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")

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
    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10):
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
        rospy.logdebug("TurtleBot3 Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        #self.wait_until_twist_achieved(cmd_vel_value,epsilon,update_rate)
        # Weplace a waitof certain amiunt of time, because this twist achived doesnt work properly
        time.sleep(0.2)

    def send_to_location(self, pt):

        move_msg = PoseStamped()
        move_msg.header.frame_id = self.robot_namespace_no_slash + '/map'

        move_msg.pose.position.x = pt[0]
        move_msg.pose.position.y = pt[1]
        move_msg.pose.position.z = 0

        move_msg.pose.orientation.x = 0
        move_msg.pose.orientation.y = 0
        move_msg.pose.orientation.z = 0
        move_msg.pose.orientation.w = 1

        self.visualize_goal_point(move_msg)

        if self.check_pose_possible(move_msg):
            print("sending to pose")
            self.move_pub.publish(move_msg)
            self.wait_until_move_achieved(move_msg)
        else:
            rospy.logwarn("Goal pose is impossible to plan to: " + str(move_msg))

    def wait_until_move_achieved(self, goal_pose):
        rospy.loginfo("waiting for move base to finish goal")

        # while not rospy.is_shutdown():
        #     curr_odom = self.get_odom()
        #     curr_x = curr_odom.pose.pose.position.x
        #     curr_y = curr_odom.pose.pose.position.y
        #
        #     goal_x = goal_pose.pose.position.x
        #     goal_y = goal_pose.pose.position.y
        #
        #     curr_map = self.get_map()
        #
        #     index_of_robot = int(self.point_to_index((curr_x, curr_y), curr_map))
        #     pt = self.index_to_point(index_of_robot, curr_map)
        #     curr_location_map_frame = self.map_to_world(pt[0], pt[1], curr_map)
        #
        #     print("CURRENT LOCATION: " + str(curr_location_map_frame))
        #     print("GOAL LOCATION: (" + str(goal_x) + ", " + str(goal_y) + ")")
        #
        #     if abs(goal_x-curr_location_map_frame[0]) <= self.move_base_threshold and abs(goal_y-curr_location_map_frame[1]) <= self.move_base_threshold:
        #         rospy.loginfo("Robot has reached desired pose")
        #         print("Robot has reached desired pose")
        #         break
        #
        #     # elif self.is_stuck():
        #     #     rospy.logerr("Robot is stuck")
        #     #     print("Robot is stuck")
        #     #     break

        self.record_move_base_response = True
        while not rospy.is_shutdown():

            if self.move_base_result is not None:
                text = self.move_base_result.status.text
                self.record_move_base_response = False
                self.move_base_result = None
                if text == "Goal reached.":
                    rospy.logwarn("GOAL REACHED!!!")
                    print("GOAL REACHED!!!")
                    break
                else:
                    rospy.logwarn("GOAL _NOT_ REACHED!!!")
                    print("GOAL _NOT_ REACHED!!!")
                    self.goal_delete_pub.publish(GoalID())
                    break

            # elif self.is_stuck():
            #     rospy.logerr("Robot is stuck")
            #     print("Robot is stuck")
            #     self.goal_delete_pub.publish(GoalID())
            #     break


    def check_pose_possible(self, goal_pose):

        move_base_service_topic = self.robot_namespace + '/move_base/make_plan'
        print("waiting for %s service", move_base_service_topic)

        rospy.wait_for_service(move_base_service_topic)

        req = GetPlan()

        curr_odom = self.get_odom()
        start_pose = PoseStamped()
        start_pose.header.frame_id = goal_pose.header.frame_id

        start_pose.pose.position.x = curr_odom.pose.pose.position.x
        start_pose.pose.position.y = curr_odom.pose.pose.position.y
        start_pose.pose.position.z = 0

        start_pose.pose.orientation.x = 0
        start_pose.pose.orientation.y = 0
        start_pose.pose.orientation.z = 0
        start_pose.pose.orientation.w = 1

        req.start = start_pose
        req.goal = goal_pose
        req.tolerance = .5

        try:
            get_plan = rospy.ServiceProxy(move_base_service_topic, GetPlan)
            resp = get_plan(req.start, req.goal, req.tolerance)
            rospy.loginfo(resp)

            if len(resp.plan.poses) > 0:
                rospy.loginfo("PATH FOUND - goal is possible to get to")
                return True
            else:
                rospy.loginfo("NO path found - goal is NOT possible to get to")
                return False

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            rospy.logerr("this means a goal is currently executing - no plan can be made right now")
            return False

    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        rospy.logdebug("START wait_until_twist_achieved...")

        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0
        epsilon = 0.05

        rospy.logdebug("Desired Twist Cmd>>" + str(cmd_vel_value))
        rospy.logdebug("epsilon>>" + str(epsilon))

        linear_speed = cmd_vel_value.linear.x
        angular_speed = cmd_vel_value.angular.z

        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon
        angular_speed_plus = angular_speed + epsilon
        angular_speed_minus = angular_speed - epsilon

        while not rospy.is_shutdown():
            current_odometry = self._check_odom_ready()
            # IN turtlebot3 the odometry angular readings are inverted, so we have to invert the sign.
            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = -1*current_odometry.twist.twist.angular.z

            rospy.logdebug("Linear VEL=" + str(odom_linear_vel) + ", ?RANGE=[" + str(linear_speed_minus) + ","+str(linear_speed_plus)+"]")
            rospy.logdebug("Angular VEL=" + str(odom_angular_vel) + ", ?RANGE=[" + str(angular_speed_minus) + ","+str(angular_speed_plus)+"]")

            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
            angular_vel_are_close = (odom_angular_vel <= angular_speed_plus) and (odom_angular_vel > angular_speed_minus)

            if linear_vel_are_close and angular_vel_are_close:
                rospy.logdebug("Reached Velocity!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rospy.logdebug("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time- start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time)+"]")

        rospy.logdebug("END wait_until_twist_achieved...")

        return delta_time


    def is_stuck(self):
        x_list = list()
        y_list = list()
        z_list = list()
        recorded_odom = self.odom_recorder
        for odom in recorded_odom:
            x_list.append(odom.pose.pose.position.x)
            y_list.append(odom.pose.pose.position.y)
            z_list.append(odom.pose.pose.position.z)

        if abs(max(x_list)-min(x_list)) < self.odom_diff_threshold and abs(max(y_list)-min(y_list)) < self.odom_diff_threshold and abs(max(z_list)-min(z_list)) < self.odom_diff_threshold:
            return True
        else:
            return False


    def get_odom(self):
        return self.odom

    def get_imu(self):
        return self.imu

    def get_laser_scan(self):
        return self.laser_scan

    def get_frontier_map(self):
        # while not self.frontier_map_receivedt:
        #     pass
        return self.frontier_map

    def get_map(self):
        # while not self.map_received:
        #     pass
        return self.map

    def get_pose_map(self):
        # while not self.pose_map_received:
        #     pass
        return self.pose_map

    def get_frontier_map_set(self):
        return self.frontier_map_received

    def get_map_set(self):
        return self.map_received

    def get_pose_map_set(self):
        return self.pose_map_received

    def pub_reset_gmapping(self):
        reset_msg = String()
        reset_msg.data = 'reset'
        self.reset_gmapping_pub.publish(reset_msg)
        print("Sent reset signal")

    def visualize_goal_point(self, marker_pose):

        marker = Marker()
        marker.action = Marker.ADD
        marker.header.frame_id = self.robot_namespace_no_slash + "/map" #self.robot_namespace_no_slash + "/base_link"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.pose.position.x = marker_pose.pose.position.x
        marker.pose.position.y = marker_pose.pose.position.y
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25

        marker.color.a = 0.5
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.visualization_pub.publish(marker)

    def point_to_index(self, point, my_map):
        """convert a index to a point"""
        pt = self.world_to_map(point[0], point[1], my_map)
        return pt[1] * my_map.info.width + pt[0]

    def convert_location(self, loc, my_map):
        """converts points to the grid"""
        res = my_map.info.resolution
        Xorigin = my_map.info.origin.position.x
        Yorigin = my_map.info.origin.position.y

        # offset from origin and then divide by resolution
        Xcell = int((loc[0] - Xorigin - (res / 2)) / res)
        Ycell = int((loc[1] - Yorigin - (res / 2)) / res)
        return (Xcell, Ycell)


    def index_to_point(self, index, my_map):
        x = index % int(my_map.info.width)
        y = (index - x) / my_map.info.width
        return (x, y)


    def map_to_world(self, x, y, my_map):
        """
            converts a point from the map to the world
            :param x: float of x position
            :param y: float of y position
            :return: tuple of converted point
        """
        resolution = my_map.info.resolution

        originX = my_map.info.origin.position.x
        originY = my_map.info.origin.position.y

        # multiply by resolution, then move by origin offset
        x = x * resolution + originX + resolution / 2
        y = y * resolution + originY + resolution / 2
        return (x, y)

    def world_to_map(self, x, y, my_map):
        """
            converts a point from the world to the map
            :param x: float of x position
            :param y: float of y position
            :return: tuple of converted point
        """

        return self.convert_location((x, y), my_map)

    def start_move_base(self):
        self.start_move_base_pub.publish("start")

    def stop_move_base(self):
        self.stop_move_base_pub.publish("stop")


