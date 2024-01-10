import os
import itertools
import threading
import sys

import rospy
import tf2_ros
import rospkg
import roslaunch
import numpy as np
from gym import spaces
import turtlebot3_two_robots_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid

# Utils
from utils.pseudo_collision_detector import PseudoCollisionDetector
from utils.image_similarity_ros import compare_current_map_to_actual_map, get_number_of_almost_white_pixels, get_number_of_almost_white_pixels_current_map
from utils.relative_movement import get_robot_position_in_map
from utils import scale, hector_path_save_publisher, simplify_occupancy_grid


class TurtleBot3WorldMapping2RobotsEnv(turtlebot3_two_robots_env.TurtleBot3TwoRobotsEnv):
    def __init__(self, yaml_config_file="turtlebot3_world_mapping.yaml"):
        """
        This Task Env is designed for having two TurtleBot3 robots in the turtlebot3 world closed room with columns.

        It will learn how to move around without crashing.
        """
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot3/ros_ws_abspath", None)
        if os.environ.get('ROS_WS') != None:
            ros_ws_abspath = os.environ.get('ROS_WS')

        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/turtlebot3_my_envs/config",
                               yaml_file_name=yaml_config_file)


        # Depending on which environment we're in, decide to launch Gazebo with or without GUI.
        gazebo_launch_file = "start_empty_tb3_world.launch"

        if os.environ.get('ENV') == 'deploy' or os.environ.get('ENV') == 'dev-no-gazebo':
            gazebo_launch_file = "start_empty_tb3_world_no_gui.launch"

        ROSLauncher(rospackage_name="coop_mapping",
                    launch_file_name=gazebo_launch_file,
                    ros_ws_abspath=ros_ws_abspath)

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot3WorldMapping2RobotsEnv, self).__init__(ros_ws_abspath,
                                                               ros_launch_file_package="coop_mapping",
                                                               ros_launch_file_name="spawn_2_robots.launch")

        ### ACTIONS
        # Only variable needed to be set here
        self.number_actions = rospy.get_param('/turtlebot3/n_actions')
        self.number_robots = len(self.robot_namespaces)  # should be 2

        # 3x3 possible actions (a % 3 -> robot 1 action, a / 3 -> robot 2 action)
        self.action_space = spaces.Discrete(
            pow(self.number_actions, self.number_robots))

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        ### OBSERVATIONS
        self.linear_forward_speed = rospy.get_param(
            '/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param(
            '/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')
        self.init_linear_forward_speed = rospy.get_param(
            '/turtlebot3/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param(
            '/turtlebot3/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/turtlebot3/new_ranges')
        self.min_range = rospy.get_param('/turtlebot3/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot3/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot3/min_laser_value')

        """
        An observation is a MultiDiscrete element, with 4 components.
        
        1. LaserScan rays component with R integers, where R is the number of laser scan 
        rays we are using for observation (one for each robot).
            - Each value represents the distance to an obstacle rounded to the nearest integer (in meteres).
            
        2. Position information with 2 integers (x, y) (one for each robot).
            - Each value represents the position of the robot along a normalized axis, rouned to the nearest integer.

        3. Rotation information with 1 integer (rotation along the z axis) (one for each robot).
            - Each value represents the orientation in a normalized scale, rounded to the nearest integer.

        4. Simplified map exploration with NxN integers, where N is the dimension of the matrix 
        that portrays the level of exploration in the map (one for BOTH robots).
            - Each value represents the average number of pixels explored (-1 is unexplored, 1 is explored). 
            The value is normalized and then rounded to the nearest integer.
        """

        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self.laser_scan[self.robot_namespaces[0]]
        num_laser_readings = len(laser_scan.ranges)/self.new_ranges
        high = np.full((num_laser_readings), self.max_laser_value)
        low = np.full((num_laser_readings), self.min_laser_value)

        self.position_num_discrete_values = 40

        self.rotation_num_discrete_values = 6

        self.simplified_grid_dimension = 4
        self.simplified_grid_num_discrete_values = 40

        laser_scan_component_shape = [
            round(self.max_laser_value)] * (self.new_ranges * self.number_robots)

        position_component_shape = [
            self.position_num_discrete_values] * (2 * self.number_robots)

        rotation_component_shape = [
            self.rotation_num_discrete_values] * (1 * self.number_robots)

        map_exploration_component_shape = [
            self.simplified_grid_num_discrete_values] * self.simplified_grid_dimension * self.simplified_grid_dimension

        multi_discrete_shape = list(itertools.chain(laser_scan_component_shape,
                                                    position_component_shape,
                                                    rotation_component_shape,
                                                    map_exploration_component_shape))

        self.observation_space = spaces.MultiDiscrete(
            multi_discrete_shape)

        rospy.loginfo("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.loginfo("OBSERVATION SPACES TYPE===>" +
                      str(self.observation_space))

        # Rewards

        self.no_crash_reward_points = rospy.get_param(
            "/turtlebot3/no_crash_reward_points")
        self.crash_reward_points = rospy.get_param(
            "/turtlebot3/crash_reward_points")
        self.exploration_multi_factor = rospy.get_param(
            "/turtlebot3/exploration_multi_factor")

        self.cumulated_steps = 0.0

        # Init dictionary for both robots actions
        self.last_action = {}

        # Set the logging system
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('coop_mapping')

        # Control variables for launching nodes from .launch files
        self._gmapping_launch_file = pkg_path + os.path.sep + \
            'launch' + os.path.sep + 'init_2_robots_mapping.launch'
        self._gmapping_running = False
        self._gmapping_launch = None

        self._map_merge_launch_file = pkg_path + os.path.sep + \
            'launch' + os.path.sep + 'init_2_robots_multi_map_merge.launch'
        self._map_merge_running = False
        self._map_merge_launch = None

        self._hector_saver_launch_file = pkg_path + os.path.sep + \
            'launch' + os.path.sep + 'init_2_robots_hector_saver.launch'
        self._hector_saver_running = False
        self._hector_saver_launch = None

        # Variables for map comparison
        self.map_data = None
        self._num_white_pixels_to_explore = get_number_of_almost_white_pixels(self.actual_map_file)

        # The minimum difference that has been observed
        self.current_min_map_difference = None

        # The thresholds to calculate map exploration
        self.threshold_occupied = 65
        self.threshold_free = 25

        # The area in pixels that has been explored
        self.previous_max_explored_area = None
        self.current_max_explored_area = None

        # Start subscriber to /map to save it to file
        self._map_file_name = "/tmp/ros_merge_map"
        self._map_updated_after_action = False
        rospy.Subscriber('map', OccupancyGrid, self._map_callback)

        # Logging episode information
        self._first_episode = True

        # TF listener to get robots position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    ### OVERRIDES

    def step(self, action):
        if os.environ.get('STEP_DEBUG') != None:
            sys.stderr.write(
                "Waiting for permission to do next step... Press a key and ENTER: ")
            raw_input()

        return super(TurtleBot3WorldMapping2RobotsEnv, self).step(action)

    def _set_init_pose(self):
        """Sets the Robots in its init pose
        """
        for ns in self.robot_namespaces:
            self.move_base(self.init_linear_forward_speed,
                           self.init_linear_turn_speed,
                           ns,
                           epsilon=0.01,
                           update_rate=10)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode (when reset is called).
        """
        self._first_episode = False

        # For Info Purposes
        self.cumulated_reward = 0.0

        # Accuracy
        self.current_min_map_difference = 1

        # Exploration
        self.previous_max_explored_area = 0
        self.current_max_explored_area = 0
        self.map_data = None

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self._crashed = False
        self._map_updated_after_action = False

        # (Re)Start GMapping, MapMerge and HectorSaver
        self._stop_gmapping()
        self._stop_map_merge()

        if os.environ.get('TEST') is not None:
            self._stop_hector_saver()

        self._start_gmapping()
        self._start_map_merge()
        
        if os.environ.get('TEST') is not None:
            self._start_hector_saver()

        # Wait for first map information and exploration result, so we don't get inflated rewards
        rate = rospy.Rate(10.0)
        while self.map_data is None:
            rate.sleep()

        self._calculate_map_exploration(self.map_data)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the two TurleBot3 based on the action value.
        """

        robot_actions = {}

        # First robot has action -> action % 3
        robot_actions[self.robot_namespaces[0]] = action % self.number_actions
        # Second robot has action -> action // 3
        robot_actions[self.robot_namespaces[1]] = action // self.number_actions

        threads = []

        for ns in self.robot_namespaces:
            current_robot_action = robot_actions[ns]
            rospy.loginfo("Start Set Action for Robot {} ==> ".format(
                ns) + str(current_robot_action))
            # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
            if current_robot_action == 0:  # FORWARD
                linear_speed = self.linear_forward_speed
                angular_speed = 0.0
                self.last_action[ns] = "FORWARDS"
            elif current_robot_action == 1:  # LEFT
                linear_speed = self.linear_turn_speed
                angular_speed = self.angular_speed
                self.last_action[ns] = "TURN_LEFT"
            elif current_robot_action == 2:  # RIGHT
                linear_speed = self.linear_turn_speed
                angular_speed = -1*self.angular_speed
                self.last_action[ns] = "TURN_RIGHT"

            # We tell TurtleBot3 the linear and angular speed to set to execute
            t = threading.Thread(target=self.move_base,
                                 args=(linear_speed, angular_speed, ns,), kwargs={"epsilon": 0.01, "update_rate": 10})

            threads.append(t)
            t.start()

            rospy.loginfo(
                "Setting Action for Robot {} ==>".format(ns)+str(action))

        for t in threads:
            t.join()

        self._map_updated_after_action = False

        rospy.loginfo("Finished bot action settings.")

    def _get_obs(self):
        """
        Here we define the observation.
        """
        rospy.loginfo("Start Get Observation ==>")
        # Set stuff for the reward calculation
        # Set the exploration values (wait for map to be available)
        rate = rospy.Rate(10.0)
        while self.map_data is None or not self._map_updated_after_action:
            rate.sleep()

        self._calculate_map_exploration(self.map_data)

        # Set the accuracy values
        # Maximum value for difference is 1. Lowest (and best) is 0.
        new_map_difference = compare_current_map_to_actual_map(
            self._map_file_name, self.actual_map_file)

        # In case the map_difference wrongfully goes up, we keep our best difference    
        self.new_min_map_difference = min(
            new_map_difference, self.current_min_map_difference)

        self.current_min_map_difference = self.new_min_map_difference

        # Now we gather the observations
        all_robots_observations = []

        for ns in self.robot_namespaces:
            # For each robot, we gather the laser_scan, position and rotation obs
            all_robots_observations.extend(self._get_laser_scan_obs(ns))
            all_robots_observations.extend(self._get_position_and_rotation_obs(ns))

        # The map_exploration obs is common to both robots
        all_robots_observations.extend(self._get_map_exploration_obs())

        rospy.loginfo("Observations from all robots==>" +
                      str(all_robots_observations))
        rospy.loginfo("END Get Observation ==>")

        return all_robots_observations

    def _is_done(self, observations):

        if self._episode_done:
            rospy.logerr("A TurtleBot3 is Too Close to wall==>")
        else:
            rospy.loginfo("No TurtleBot3 is close to a wall ==>")

        self._save_map_image(self.map_data)

        estimated_white_pixels = get_number_of_almost_white_pixels_current_map(self._map_file_name)

        # If the robot has mapped a percentage of the estimated area
        area_percentage = 0.85
        if estimated_white_pixels >= self._num_white_pixels_to_explore * area_percentage:
            self._episode_done = True
            rospy.logerr("Turtlebots have mapped {} of the area.".format(area_percentage))

        rospy.logwarn("Turtlebots have explored {} pixels out of {} (a {} ratio).".format(estimated_white_pixels,
                                                                                          self._num_white_pixels_to_explore, 
                                                                                          estimated_white_pixels * 1.0 / self._num_white_pixels_to_explore))

        if self._episode_done:
            self.save_episode_info()

        return self._episode_done

    def _compute_reward(self, observations, done):
        """
        Compute reward. The step reward is a value between -1 and 1.
        """
        rospy.logwarn("Running map comparison...")

        # If we decrease the map difference, it should be rewarded.
        accuracy_reward_base = self.current_min_map_difference
        accuracy_reward_weight = 1000.0

        # Maximum possible explored area is the area of our map, so we normalize what we have explored, to be between 0 and 1.
        area_reward_base = (self.current_max_explored_area - self.previous_max_explored_area) * 1.0 # Number of explored pixels
        area_reward_weight = 1.0

        if not done:
            reward = area_reward_base * area_reward_weight - self.no_crash_reward_points
        else:
            if self._crashed:
                reward = - self.crash_reward_points
            else:
                reward = accuracy_reward_base * accuracy_reward_weight

        rospy.loginfo("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.loginfo("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.loginfo("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    # Internal TaskEnv Methods

    def check_if_crashed(self, ns):
        """
        Check if any of the robots has crashed. 
        """
        laser_scan_message = self.laser_scan[ns]

        collision_detector = PseudoCollisionDetector()

        collision_detected = collision_detector.collision_detected(
            laser_scan_message, self.min_range)

        if collision_detected:
            return True

        return False

    def _start_map_merge(self):
        if not self._map_merge_running:
            rospy.loginfo("Creating launch parent for MapMerge launch file.")
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)

            self._map_merge_launch = roslaunch.parent.ROSLaunchParent(
                uuid, [self._map_merge_launch_file])

            self._map_merge_launch.start()
            rospy.loginfo("Started MapMerge launch file.")

            self._map_merge_running = True

    def _start_gmapping(self):
        if not self._gmapping_running:
            rospy.loginfo("Creating launch parent for Gmapping launch file.")
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            self._gmapping_launch = roslaunch.parent.ROSLaunchParent(
                uuid, [self._gmapping_launch_file])

            self._gmapping_launch.start()
            rospy.loginfo("Started Gmapping launch file.")

            self._gmapping_running = True

    def _start_hector_saver(self):
        if not self._hector_saver_running:
            rospy.loginfo(
                "Creating launch parent for Hector Saver launch file.")
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            self._hector_saver_launch = roslaunch.parent.ROSLaunchParent(
                uuid, [self._hector_saver_launch_file])

            self._hector_saver_launch.start()
            rospy.loginfo("Started Hector Saver launch file.")

            self._hector_saver_running = True

    def _stop_map_merge(self):
        if self._map_merge_running:
            self._map_merge_launch.shutdown()
            rospy.loginfo("Stopped MapMerge launch file.")

            self._map_merge_running = False

    def _stop_gmapping(self):
        if self._gmapping_running:
            self._gmapping_launch.shutdown()
            rospy.loginfo("Stopped Gmapping launch file.")

            self._gmapping_running = False

    def _stop_hector_saver(self):
        if self._hector_saver_running:
            self._hector_saver_launch.shutdown()
            rospy.loginfo("Stopped Hector Saver launch file.")

            self._hector_saver_running = False

    def _map_callback(self, map_data):
        # Based on this: https://github.com/ros-planning/navigation/blob/melodic-devel/map_server/src/map_saver.cpp

        # Save map_data
        self.map_data = map_data
        self._map_updated_after_action = True

    ### OBSERVATION-RELATED METHODS

    def _discretize_laser_scan_observation(self, data, new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        discretized_ranges = []
        mod = len(data.ranges)/new_ranges

        # rospy.loginfo("data=" + str(data))
        rospy.loginfo("new_ranges=" + str(new_ranges))
        rospy.loginfo("mod=" + str(mod))

        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                if item == float('Inf') or np.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif np.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))

                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" +
                                str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.loginfo("NOT done Validation >>> item=" +
                                str(item)+"< "+str(self.min_range))

        return discretized_ranges

    def _discretize_position_and_rotation_observation(self, position, rotation):
        if position is None:
            position = [0, 0, 0]
        if rotation is None:
            rotation = [0, 0, 0]
        pos_x, pos_y, pos_z = position  # in meters
        rot_x, rot_y, rot_z = rotation  # in radians

        # We are only interested in pos_x and pos_y (pos_z should be constant)
        # We rescale both values from an assumed scale of -10 to 10 (we clip them), to a new scale 0 to (max-min) values
        clipped_pos_x, clipped_pos_y = np.clip([pos_x, pos_y], -10, 10)
        scaled_pos_x = scale.scale_scalar(
            clipped_pos_x, -10, 10, 0, self.position_num_discrete_values)
        scaled_pos_y = scale.scale_scalar(
            clipped_pos_y, -10, 10, 0, self.position_num_discrete_values)

        # And we round the values
        discretized_pos_x = np.rint(scaled_pos_x)
        discretized_pos_y = np.rint(scaled_pos_y)

        # We are only interested in rot_z (rot_x and rot_y should be constant).
        # We convert it so that it gives us the positive angle of rotation (between 0 and 2pi).
        rot_z = rot_z % (2 * np.pi)

        # Now we rescale it to a scale from 0 to (max-min) values.
        scaled_rot_z = scale.scale_scalar(
            rot_z, 0, 2*np.pi, 0, self.rotation_num_discrete_values)

        # And we round it
        discretized_rot_z = np.rint(scaled_rot_z)

        return [discretized_pos_x, discretized_pos_y, discretized_rot_z]

    def _discretize_map_exploration_observation(self, map_data):
        # We already do some work in the "simplify_occupancy_grid" function.
        # Here we just want to rescale and round the values to fit our observation space.
        simplified_map_data = simplify_occupancy_grid.simplify_occupancy_grid(
            map_data, self.simplified_grid_dimension)

        scaled_map_data = scale.scale_arr(
            simplified_map_data, 0, 1, 0, self.simplified_grid_num_discrete_values)
        discretized_map_data = np.rint(scaled_map_data)

        return discretized_map_data.tolist()

    def _get_laser_scan_obs(self, namespace):
        laser_scan = self.laser_scan[namespace]
        return self._discretize_laser_scan_observation(laser_scan, self.new_ranges)

    def _get_position_and_rotation_obs(self, namespace):
        position, rotation = get_robot_position_in_map(self.tf_buffer, namespace)
        return self._discretize_position_and_rotation_observation(position, rotation)

    def _get_map_exploration_obs(self):
        rate = rospy.Rate(10.0)
        while(self.map_data is None):
            rate.sleep()    # Wait for map data to become available

        return self._discretize_map_exploration_observation(self.map_data)

    ### REWARD RELATED METHODS

    def _calculate_map_exploration(self, map_data):
        """
        Arguments:
            map_data {[type]} -- [description]
        """
        explored_area = 0

        for y in range(map_data.info.height):
            for x in range(map_data.info.width):
                i = x + (map_data.info.height - y - 1) * map_data.info.width

                if map_data.data[i] >= 0:
                    explored_area += 1

        self.previous_max_explored_area = self.current_max_explored_area

        if explored_area >= self.current_max_explored_area:
            self.current_max_explored_area = explored_area

    ### LOGGING RELATED METHODS

    def save_episode_info(self):

        # TODO: fix this
        # **WARNING**: hector_path_save must allow some time for the ROS service to execute
        # careful when moving this around, to keep the sleep

        if os.environ.get('TEST') is not None:
            # Save trajectory of the robots
            for ns in self.robot_namespaces:
                hector_path_save_publisher.publish_once(ns)

            # Becaseu hector_saver needs some time to run, before we delete its nodes
            rospy.sleep(1)

        # Save current map representation as an image
        self._save_map_image(self.map_data)

    def _save_map_image(self, map_data):
        # Open a tmp file to avoid racing condition
        f = open(self._map_file_name + "_tmp.pgm", "w")

        f.write("P5\n# CREATOR: turtlebot3_world_mapping_2_robots.py {} m/pix\n{} {}\n255\n".format(map_data.info.resolution,
                                                                                                    map_data.info.width,
                                                                                                    map_data.info.height))

        for y in range(map_data.info.height):
            for x in range(map_data.info.width):
                i = x + (map_data.info.height - y - 1) * map_data.info.width

                if map_data.data[i] >= 0 and map_data.data[i] <= self.threshold_free:
                    f.write(chr(254))

                elif map_data.data[i] >= self.threshold_occupied:
                    f.write(chr(0))

                else:
                    f.write(chr(205))

        f.close()

        # Rename file if possible.
        os.rename(self._map_file_name + "_tmp.pgm",
                  self._map_file_name + ".pgm")
