import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import turtlebot3_env_custom_a2c
from gym.envs.registration import register
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Point, PoseStamped, Pose
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from openai_ros.openai_ros_common import ROSLauncher
import rosnode
import os
import time


class TurtleBot3WorldEnv(turtlebot3_env_custom_a2c.TurtleBot3Env):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot3/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        # ROSLauncher(rospackage_name="turtlebot3_gazebo",
        #             launch_file_name="start_world.launch",
        #             ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/turtlebot3_custom_a2c/config",
                               yaml_file_name="turtlebot3_world.yaml")


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot3WorldEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot3/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)


        #number_observations = rospy.get_param('/turtlebot3/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot3/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot3/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/turtlebot3/new_ranges')
        self.min_range = rospy.get_param('/turtlebot3/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot3/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot3/min_laser_value')
        self.max_linear_aceleration = rospy.get_param('/turtlebot3/max_linear_aceleration')


        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        # laser_scan = self.get_laser_scan()
        # rospy.logwarn("LASER SCAN " + str(laser_scan))
        # num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)
        # high = numpy.full((num_laser_readings), self.max_laser_value)
        # low = numpy.full((num_laser_readings), self.min_laser_value)
        #
        # # We only use two integers
        # self.observation_space = spaces.Box(low, high)
        number_observations = rospy.get_param('/turtlebot3/n_observations')
        self.observation_space = spaces.Discrete(number_observations)


        print("ACTION SPACES TYPE===>"+str(self.action_space))
        print("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        # self.forwards_reward = 5 #rospy.get_param("/turtlebot3/forwards_reward")
        # self.turn_reward = 5 #rospy.get_param("/turtlebot3/turn_reward")
        # self.end_episode_points = 200 #rospy.get_param("/turtlebot3/end_episode_points")

        self.unoccupied_reward = rospy.get_param("/turtlebot3/unoccupied_reward")
        self.unknown_reward = rospy.get_param("/turtlebot3/unknown_reward")
        self.occupied_reward = rospy.get_param("/turtlebot3/occupied_reward")
        self.end_episode_points = rospy.get_param("/turtlebot3/end_episode_points")
        self.open_cell_found_points = rospy.get_param("/turtlebot3/open_cell_found_points")
        self.occupied_cell_found_points = rospy.get_param("/turtlebot3/occupied_cell_found_points")
        self.travel_time_points = rospy.get_param("/turtlebot3/travel_time_points")
        self.near_wall_points = rospy.get_param("/turtlebot3/near_wall_points")

        self.cumulated_steps = 0.0

        self.first_ep = True

        # self.episode_start_time = rospy.get_time()
        self.episode_start_time = time.perf_counter()
        self.max_ep_time = 2700 #5 #1000 # 1800 * (3/60)


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        print("ACTION: " + str(action))

        while not self.get_map_set():
            rospy.logwarn("Map not set yet - waiting one second")
            rospy.sleep(1)
        print("Map set - continuing with choosing action")

        map = self.get_map()

        if self.is_valid_point(action, map):
            pt = self.index_to_point(action, map)

            if self.get_occupied_neighbors((pt[0],pt[1]), map):
                rospy.logwarn("NOT SENDING TO POINT BECAUSE NEXT TO WALL")

            else:
                real_point = self.map_to_world(pt[0], pt[1], map)

                rospy.logwarn("SENDING TO POINT: " + str(real_point))

                self.send_to_location(real_point)

        else:
            rospy.logwarn("ACTION INDEX POINT IS NOT VALID - MOVING ON")


        rospy.loginfo("END Set Action ==> "+ str(action))
        self.last_action = action
        self.last_action_start_time = rospy.get_time()

        # rospy.logdebug("Start Set Action ==>"+str(action))
        # action = 0
        # # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        # if action == 0: #FORWARD
        #     linear_speed = self.linear_forward_speed
        #     angular_speed = 0.0
        #     self.last_action = "FORWARDS"
        # elif action == 1: #LEFT
        #     linear_speed = self.linear_turn_speed
        #     angular_speed = self.angular_speed
        #     self.last_action = "TURN_LEFT"
        # elif action == 2: #RIGHT
        #     linear_speed = self.linear_turn_speed
        #     angular_speed = -1*self.angular_speed
        #     self.last_action = "TURN_RIGHT"
        #
        # # We tell TurtleBot2 the linear and angular speed to set to execute
        # self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        #
        # rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        # rospy.logdebug("Start Get Observation ==>")
        # # We get the laser scan data
        # laser_scan = self.get_laser_scan()
        #
        # discretized_observations = self.discretize_scan_observation(laser_scan, self.new_ranges)
        #
        # print("Observations==>"+str(discretized_observations))
        # print("END Get Observation ==>")
        #
        # return discretized_observations

        if not self.get_map_set() or not self.get_frontier_map_set() or not self.get_pose_map_set():

            rospy.logwarn("Observations all made occupied out because not set yet")
            # obs = numpy.zeros(self.observation_space.n)
            obs = numpy.full(self.observation_space.n, 100)
            return obs

        else:

            print("Observation maps set - continuing with creating observations")

            observation1 = self.get_frontier_map()
            observation2 = self.get_map()
            observation3 = self.get_pose_map()

            full_obs = numpy.array(observation1.data)
            full_obs = numpy.append(full_obs, numpy.array(observation2.data))
            full_obs = numpy.append(full_obs, numpy.array(observation3.data))

            return full_obs


    def get_obs(self):
        return self._get_obs()


    def _is_done(self, observations):

        if self._episode_done:
            rospy.logerr("Turtlebot is ending episode")
        else:
            rospy.logwarn("TurtleBot is continuing with episode")

        # Now we check if it has crashed based on the imu
        imu_data = self.get_imu()
        linear_acceleration_magnitude = self.get_vector_magnitude(imu_data.linear_acceleration)
        if linear_acceleration_magnitude > self.max_linear_aceleration:
            rospy.logerr("TurtleBot2 Crashed==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))
            self._episode_done = True
        else:
            rospy.logerr("DIDNT crash TurtleBot2 ==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))

        if self.get_frontier_map_set():
            fringe_data = numpy.array(self.get_frontier_map().data)
            if 0 not in fringe_data:
                rospy.logerr("Turtlebot has no frontiers left ====>")
                self._episode_done = True

        # print("Time left in episode: " + str(self.max_ep_time - (rospy.get_time() - self.episode_start_time)) + " seconds")
        # if rospy.get_time() - self.episode_start_time > self.max_ep_time:
        #     rospy.logerr("Turtlebot is ending episode due to max time being reached")
        #     self._episode_done = True

        print("Time left in episode: " + str(self.max_ep_time - (time.perf_counter() - self.episode_start_time)) + " seconds")
        if time.perf_counter() - self.episode_start_time > self.max_ep_time:
            rospy.logerr("Turtlebot is ending episode due to max time being reached")
            self._episode_done = True

        return self._episode_done


    def _compute_reward(self, observations, done):

        if self.first_ep:
            reward = -1 * self.end_episode_points
            if self.get_map_set():
                self.last_computed_reward_map = numpy.array(self.get_map().data)
            else:
                self.last_computed_reward_map = numpy.full(self.observation_space.n/3, 100)
            self.first_ep = False

        else:
            if not done and self.get_map_set():
                map = self.get_map()
                data = map.data

                occupancy = data[int(self.last_action)]
                if occupancy == -1:
                    reward = self.unknown_reward
                elif occupancy == 0:
                    reward = self.unoccupied_reward
                elif occupancy > 0:
                    # penalty
                    reward = -1 * self.occupied_reward

                n_open_prev_map = numpy.count_nonzero(self.last_computed_reward_map == 0)
                n_open_this_map = numpy.count_nonzero(numpy.array(data) == 0)

                n_occupied_prev_map = numpy.count_nonzero(self.last_computed_reward_map == 100)
                n_occupied_this_map = numpy.count_nonzero(numpy.array(data) == 100)

                amount_of_open_cells_discovered = n_open_this_map - n_open_prev_map
                amount_of_occupied_cells_discovered = n_occupied_this_map - n_occupied_prev_map

                reward += amount_of_open_cells_discovered*self.open_cell_found_points + amount_of_occupied_cells_discovered*self.occupied_cell_found_points

                # penalty
                time_reward = -1 * (rospy.get_time() - self.last_action_start_time) * self.travel_time_points
                reward += time_reward

                print("time reward: " + str(time_reward))

                # penalty
                pt = self.index_to_point(self.last_action, map)
                if self.get_occupied_neighbors((pt[0], pt[1]), map):
                    near_wall_points = -1 * self.near_wall_points
                    reward += near_wall_points
                    print("near_wall reward: " + str(near_wall_points))

                self.last_computed_reward_map = numpy.array(data)

            else:
                reward = -1*self.end_episode_points

        # if not done:
        #     if self.last_action == "FORWARDS":
        #         reward = self.forwards_reward
        #     else:
        #         reward = self.turn_reward
        # else:
        #     reward = -1 * self.end_episode_points


        rospy.loginfo("reward=" + str(reward))
        print("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.loginfo("Cumulated_reward=" + str(self.cumulated_reward))
        print("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.loginfo("Cumulated_steps=" + str(self.cumulated_steps))
        print("Cumulated_steps=" + str(self.cumulated_steps))

        return reward


    # Internal TaskEnv Methods

    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges)/new_ranges

        rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))

        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))

                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logdebug("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))


        return discretized_ranges


    def get_vector_magnitude(self, vector):
        """
        It calculated the magnitude of the Vector3 given.
        This is usefull for reading imu accelerations and knowing if there has been
        a crash
        :return:
        """
        contact_force_np = numpy.array((vector.x, vector.y, vector.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude


    def index_to_point(self, index, my_map):
        x = index % int(my_map.info.width)
        y = (index - x)/my_map.info.width
        return (x,y)


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


    def is_valid_point(self, index, map):
        data = map.data
        if data is not None and len(data) >= index:
            if data[index] > 95 or data[index] < 0:
                # print("occupied or unknown space")
                return False
            else:
                return True
        else:
            # print("data is none")
            return False

    def is_occupied_point(self, index, map):
        data = map.data
        if data is not None and len(data) >= index:
            if data[index] > 95:
                print("occupied space")
                return True
            else:
                return False
        else:
            # print("data is none")
            return False

    def get_occupied_neighbors(self, loc, my_map):

        # my_map: http://docs.ros.org/melodic/api/nav_msgs/html/msg/OccupancyGrid.html
        """
            returns the legal neighbors of loc
            :param loc: tuple of location
            :return: list of tuples
        """
        # get x and y coordinates
        Xloc = loc[0]
        Yloc = loc[1]

        # relative locations to look at
        neighbors = list()
        potentNeighbors = []
        case1 = (Xloc + 1, Yloc)
        case2 = (Xloc - 1, Yloc)
        case3 = (Xloc, Yloc + 1)
        case4 = (Xloc, Yloc - 1)

        # these are the nodes we want to check
        potentNeighbors.append(case1)
        potentNeighbors.append(case2)
        potentNeighbors.append(case3)
        potentNeighbors.append(case4)

        # only return valid locations (i.e. not walls)
        for neighbor in potentNeighbors:
            neighbor_index = int(neighbor[1] * my_map.info.width + neighbor[0])
            if neighbor_index >= my_map.info.width * my_map.info.height:
                continue
            elif self.is_occupied_point(neighbor_index, my_map):
                neighbors.append(neighbor)

        return neighbors


    def reset_gmapping(self):
        print("RESETTING GMAPPING")
        self.pub_reset_gmapping()
        self.first_ep = True

        # while not self.gmapping_fully_reset:
        #     pass
        # print("RESETTING GMAPPING FINALIZED")
        # self.gmapping_fully_reset = False

    def set_start_time(self):
        # self.episode_start_time = rospy.get_time()
        self.episode_start_time = time.perf_counter()






