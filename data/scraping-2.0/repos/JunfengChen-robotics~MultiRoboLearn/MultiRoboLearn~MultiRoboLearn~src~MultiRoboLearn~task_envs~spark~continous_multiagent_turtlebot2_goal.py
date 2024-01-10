import rospy
import numpy
import time
import math
from gym import spaces
from openai_ros.robot_envs import multiagent_turtlebot2_env
from gym.envs.registration import register
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from geometry_msgs.msg import Point

# The path is __init__.py of openai_ros, where we import the TurtleBot2MazeEnv directly
max_episode_steps_per_episode = 100 # Can be any Value

register(
        id='MultiagentTurtleBot2-v1',
        entry_point='openai_ros.task_envs.turtlebot2.continous_multiagent_turtlebot2_goal:MultiagentTurtleBot2Env',
        max_episode_steps=max_episode_steps_per_episode,
    )

class MultiagentTurtleBot2Env(multiagent_turtlebot2_env.MultiagentTurtleBot2Env):
    def __init__(self):
        """
        This Task Env is designed for having the multi TurtleBot2 in some kind of scenarios.
        It will learn how to move around the desired point without crashing into static and dynamic obstacle.
        """
        
        # Only variable needed to be set here
        self.number_actions = rospy.get_param('/turtlebot2/n_actions')

        high = numpy.full((self.number_actions), 1.0)
        low = numpy.full((self.number_actions), -1.0)
        self.action_space = spaces.Box(low, high)

        # Maximum linear velocity (m/s) of Spark
        max_lin_vel = 0.4
        # Maximum angular velocity (rad/s) of Spark
        max_ang_vel = 0.2
        self.max_vel = numpy.array([max_lin_vel, max_ang_vel])
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        
        # Actions and Observations
        self.dec_obs = rospy.get_param("/turtlebot2/number_decimals_precision_obs", 3)
        self.linear_forward_speed = rospy.get_param('/turtlebot2/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot2/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot2/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot2/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot2/init_linear_turn_speed')
        
        
        self.n_observations = rospy.get_param('/turtlebot2/n_observations')
        self.min_range = rospy.get_param('/turtlebot2/min_range')
        # self.new_ranges = rospy.get_param('/turtlebot2/new_ranges')
        self.max_laser_value = rospy.get_param('/turtlebot2/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot2/min_laser_value')

        # Get Desired Point to Get for different robots
        # for marobot1
        self.marobot1_desired_point = Point()
        self.marobot1_desired_point.x = rospy.get_param("/turtlebot2/marobot1/desired_pose/x")
        self.marobot1_desired_point.y = rospy.get_param("/turtlebot2/marobot1/desired_pose/y")
        self.marobot1_desired_point.z = rospy.get_param("/turtlebot2/marobot1/desired_pose/z")

        self.marobot1_obstacle_point = Point()
        self.marobot1_obstacle_point.x = rospy.get_param("/turtlebot2/obstacle1/obstacle_pose/x")
        self.marobot1_obstacle_point.y = rospy.get_param("/turtlebot2/obstacle1/obstacle_pose/y")


        # for marobot2
        self.marobot2_desired_point = Point()
        self.marobot2_desired_point.x = rospy.get_param("/turtlebot2/marobot2/desired_pose/x")
        self.marobot2_desired_point.y = rospy.get_param("/turtlebot2/marobot2/desired_pose/y")
        self.marobot2_desired_point.z = rospy.get_param("/turtlebot2/marobot2/desired_pose/z")

        self.marobot2_obstacle_point = Point()
        self.marobot2_obstacle_point.x = rospy.get_param("/turtlebot2/obstacle2/obstacle_pose/x")
        self.marobot2_obstacle_point.y = rospy.get_param("/turtlebot2/obstacle2/obstacle_pose/y")

        # for marobot3
        self.marobot3_desired_point = Point()
        self.marobot3_desired_point.x = rospy.get_param("/turtlebot2/marobot3/desired_pose/x")
        self.marobot3_desired_point.y = rospy.get_param("/turtlebot2/marobot3/desired_pose/y")
        self.marobot3_desired_point.z = rospy.get_param("/turtlebot2/marobot3/desired_pose/z")

        self.marobot3_obstacle_point = Point()
        self.marobot3_obstacle_point.x = rospy.get_param("/turtlebot2/obstacle3/obstacle_pose/x")
        self.marobot3_obstacle_point.y = rospy.get_param("/turtlebot2/obstacle3/obstacle_pose/y")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(MultiagentTurtleBot2Env, self).__init__()
        
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.

        laser_scans = self.get_laser_scan()
        rospy.logdebug("laser_scan len===>"+str(len(laser_scans[0].ranges)))
        
        # Laser data for different robots
        self.laser_scan_frame_1 = laser_scans[0].header.frame_id
        self.laser_scan_frame_2 = laser_scans[1].header.frame_id
        self.laser_scan_frame_3 = laser_scans[2].header.frame_id

        
        
        # Number of laser reading jumped
        self.new_ranges = int(math.ceil(float(len(laser_scans[0].ranges)) / float(self.n_observations)))
        # self.new_ranges = 1

        rospy.logdebug("n_observations===>"+str(self.n_observations))
        rospy.logdebug("new_ranges, jumping laser readings===>"+str(self.new_ranges))
        
        
        high = numpy.full((self.n_observations), self.max_laser_value)
        #in order to validate the observation data, we modify the min_laser_value into -self.max_laser_value as low
        low = numpy.full((self.n_observations), -1*self.max_laser_value)
        # low = numpy.full((self.n_observations), self.min_laser_value)
        
        # We only use two integers
        self.observation_space = spaces.Box(low, high, dtype=numpy.float32)
        
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        #done for all robots
        # self._episode_dones = []
        
        # Rewards
        # self.forwards_reward = rospy.get_param("/turtlebot2/forwards_reward")
        # self.turn_reward = rospy.get_param("/turtlebot2/turn_reward")
        # self.end_episode_points = rospy.get_param("/turtlebot2/end_episode_points")

        self.cumulated_steps = 0.0

        self.laser_filtered_pub_1 = rospy.Publisher('marobot1/turtlebot2/laser/scan_filtered', LaserScan, queue_size=10)
        self.laser_filtered_pub_2 = rospy.Publisher('marobot2/turtlebot2/laser/scan_filtered', LaserScan, queue_size=10)
        self.laser_filtered_pub_3 = rospy.Publisher('marobot3/turtlebot2/laser/scan_filtered', LaserScan, queue_size=10)

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base_1( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10,
                        min_laser_distance=-1)

        self.move_base_2(self.init_linear_forward_speed,
                         self.init_linear_turn_speed,
                         epsilon=0.05,
                         update_rate=10,
                         min_laser_distance=-1)

        self.move_base_3(self.init_linear_forward_speed,
                         self.init_linear_turn_speed,
                         epsilon=0.05,
                         update_rate=10,
                         min_laser_distance=-1)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes,and total reward for all robots
        self.cumulated_reward = 0.0 #This only is put here, in fact, it is less useful.
        # self.cumulated_episode_reward = [0, 0, 0]

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self._episode_dones = [False,False,False]
        self._if_dones_label = [False,False,False]

        
        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)
        
        # TODO: Add reset of published filtered laser readings
        #add
        laser_scans = self.get_laser_scan()
        # laser_scans = self.get_laser_scan_spark()
        print("laser for real robots", laser_scans)
        discretized_ranges = [laser_scans[0].ranges,laser_scans[1].ranges,laser_scans[2].ranges]
        #publish different laser data for all robots
        pub_num_marobot1 = '1'
        self.publish_filtered_laser_scan(laser_original_data=laser_scans[0],
                                         new_filtered_laser_range=discretized_ranges[0],
                                         pub_num=pub_num_marobot1)

        pub_num_marobot2 = '2'
        self.publish_filtered_laser_scan(laser_original_data=laser_scans[1],
                                         new_filtered_laser_range=discretized_ranges[1],
                                         pub_num=pub_num_marobot2)

        pub_num_marobot3 = '3'
        self.publish_filtered_laser_scan(laser_original_data=laser_scans[2],
                                         new_filtered_laser_range=discretized_ranges[2],
                                         pub_num=pub_num_marobot3)

        #add
        odometrys = self.get_odom()
        # odometrys = self.get_odom_spark()
        print("odom for real robots", odometrys)
        # print("odometrys is:", odometrys)

        #add

        # odometrys[0].pose.pose.position.x = odometrys[0].pose.pose.position.x + 1
        # odometrys[0].pose.pose.position.y = odometrys[0].pose.pose.position.y + 1
        #
        #
        # # for marobot2:
        #
        #
        # odometrys[1].pose.pose.position.x = odometrys[1].pose.pose.position.x + 4
        # odometrys[1].pose.pose.position.y = odometrys[1].pose.pose.position.y + 2
        #
        #
        # # for marobot3:
        #
        #
        # odometrys[2].pose.pose.position.x = odometrys[2].pose.pose.position.x + 1
        # odometrys[2].pose.pose.position.y = odometrys[2].pose.pose.position.y + 3



        self.previous_distance_from_des_points = [self.get_distance_from_desired_point_1(odometrys[0].pose.pose.position),self.get_distance_from_desired_point_2(odometrys[1].pose.pose.position),self.get_distance_from_desired_point_3(odometrys[2].pose.pose.position)]


    def _set_action(self, actions):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """


        #for marobot1:
        action = actions[0]
        action = numpy.array(action)
        rospy.logdebug("Start Set Action for marobot1==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
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
        # elif action == 3: #BACKFORWARD
        #     linear_speed = -1*self.linear_forward_speed
        #     angular_speed = 0.0
        #     self.last_action = "BACKFORWARD"
        # elif action == 4: #STOP
        #     linear_speed = 0.0
        #     angular_speed = 0.0
        #     self.last_action = "STOP"

        action = numpy.multiply(action, self.max_vel)
        action_excution = action.tolist()
        print("agent0 action is:", action_excution)
        
        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base_1( linear_speed = action_excution[0],
                        angular_speed = action_excution[1],
                        epsilon=0.05,
                        update_rate=10,
                        min_laser_distance=self.min_range)
        
        # rospy.logdebug("END Set Action for marobot1==>"+str(action)+", NAME="+str(self.last_action))

        # for marobot2:
        action = actions[1]
        action = numpy.array(action)
        rospy.logdebug("Start Set Action for marobot1==>" + str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        # if action == 0:  # FORWARD
        #     linear_speed = self.linear_forward_speed
        #     angular_speed = 0.0
        #     self.last_action = "FORWARDS"
        # elif action == 1:  # LEFT
        #     linear_speed = self.linear_turn_speed
        #     angular_speed = self.angular_speed
        #     self.last_action = "TURN_LEFT"
        # elif action == 2:  # RIGHT
        #     linear_speed = self.linear_turn_speed
        #     angular_speed = -1 * self.angular_speed
        #     self.last_action = "TURN_RIGHT"
        # elif action == 3:  # BACKFORWARD
        #     linear_speed = -1 * self.linear_forward_speed
        #     angular_speed = 0.0
        #     self.last_action = "BACKFORWARD"
        # elif action == 4:  # STOP
        #     linear_speed = 0.0
        #     angular_speed = 0.0
        #     self.last_action = "STOP"

        action = numpy.multiply(action, self.max_vel)
        action_excution = action.tolist()

        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base_2(linear_speed = action_excution[0],
                         angular_speed = action_excution[1],
                         epsilon=0.05,
                         update_rate=10,
                         min_laser_distance=self.min_range)

        # rospy.logdebug("END Set Action for marobot2==>" + str(action) + ", NAME=" + str(self.last_action))

        # for marobot3:
        action = actions[2]
        action = numpy.array(action)
        rospy.logdebug("Start Set Action for marobot1==>" + str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        # if action == 0:  # FORWARD
        #     linear_speed = self.linear_forward_speed
        #     angular_speed = 0.0
        #     self.last_action = "FORWARDS"
        # elif action == 1:  # LEFT
        #     linear_speed = self.linear_turn_speed
        #     angular_speed = self.angular_speed
        #     self.last_action = "TURN_LEFT"
        # elif action == 2:  # RIGHT
        #     linear_speed = self.linear_turn_speed
        #     angular_speed = -1 * self.angular_speed
        #     self.last_action = "TURN_RIGHT"
        # elif action == 3:  # BACKFORWARD
        #     linear_speed = -1 * self.linear_forward_speed
        #     angular_speed = 0.0
        #     self.last_action = "BACKFORWARD"
        # elif action == 4:  # STOP
        #     linear_speed = 0.0
        #     angular_speed = 0.0
        #     self.last_action = "STOP"

        action = numpy.multiply(action, self.max_vel)
        action_excution = action.tolist()


        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base_3(linear_speed = action_excution[0],
                         angular_speed = action_excution[1],
                         epsilon=0.05,
                         update_rate=10,
                         min_laser_distance=self.min_range)

        # rospy.logdebug("END Set Action for marobot3==>" + str(action) + ", NAME=" + str(self.last_action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data for all robots

        #add
        laser_scans = self.get_laser_scan()
        # laser_scans = self.get_laser_scan_spark()


        rospy.logdebug("BEFORE DISCRET _episode_done==>"+str(self._episode_done))

        #discretize laser date for different robots:
        discretized_laser_scan_1 = self.discretize_observation( laser_scans[0],
                                                                self.new_ranges,
                                                                '1'
                                                                )
        discretized_laser_scan_2 = self.discretize_observation(laser_scans[1],
                                                               self.new_ranges,
                                                               '2'
                                                               )
        discretized_laser_scan_3 = self.discretize_observation(laser_scans[2],
                                                               self.new_ranges,
                                                               '3'
                                                               )

        # obtain laser data for all robots
        discretized_laser_scan = [discretized_laser_scan_1, discretized_laser_scan_2, discretized_laser_scan_3]
        # We get the odometry for all robots so that SumitXL knows where it is.


        #add
        odometrys = self.get_odom()
        # odometrys = self.get_odom_spark()


        #for marobot1:
        odometry = odometrys[0]
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y

        # We round to only two decimals to avoid very big Observation space
        odometry_array_1 = [round(x_position, 2), round(y_position, 2)]

        # for marobot2:
        odometry = odometrys[1]
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y

        # We round to only two decimals to avoid very big Observation space
        odometry_array_2 = [round(x_position, 2), round(y_position, 2)]

        # for marobot3:
        odometry = odometrys[2]
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y

        # We round to only two decimals to avoid very big Observation space
        odometry_array_3 = [round(x_position, 2), round(y_position, 2)]

        # for all odometry_array for all robots
        odometry_array = [odometry_array_1, odometry_array_2, odometry_array_3]


        # print("里程计数据：",odometry_array_1)

        # We only want the X and Y position and the Yaw

        # observations = discretized_laser_scan + odometry_array
        observations = []
        # observations = observations.append(discretized_laser_scan)
        # observations = observations.append(odometry_array)
        observations_marobot1 = [odometry_array[0],discretized_laser_scan[0]]
        observations_marobot2 = [odometry_array[1],discretized_laser_scan[1]]
        observations_marobot3 = [odometry_array[2],discretized_laser_scan[2]]
        observations =[observations_marobot1,observations_marobot2,observations_marobot3]

        # observations = discretized_laser_scan + odometry_array

        rospy.logdebug("Observations==>" + str(observations))
        rospy.logdebug("END Get Observation ==>")

        # rospy.logdebug("Observations==>"+str(discretized_observations))
        # rospy.logdebug("AFTER DISCRET_episode_done==>"+str(self._episode_done))
        # rospy.logdebug("END Get Observation ==>")
        return observations
        

    def _is_done(self, observations):
        #deciede per agent done and store in list

        sub_episode_done = []
        
        # if self._episode_done:
        # done[0] and done[1] and done[2]
        # if self._episode_dones[0] is False and self._episode_dones[1] is False and self._episode_dones[2] is False:
        if self._episode_dones[0] is True and self._episode_dones[1] is True and self._episode_dones[2] is True:
            # rospy.logdebug("All TurtleBot2 robots are Too Close or has crashed==>"+str(self._episode_done))
            print("All TurtleBot2 robots are Too Close or has crashed==>"+str(self._episode_dones))
            return self._episode_dones
        else:
            rospy.logerr("All TurtleBot2 robots are Ok ==>")

            current_position_1 = Point()
            current_position_2 = Point()
            current_position_3 = Point()
            # current_position.x = observations[-2]
            # current_position.y = observations[-1]
            #for marobot1:
            current_position_1.x = observations[0][0][0]
            current_position_1.y = observations[0][0][1]
            current_position_1.z = 0.0

            # for marobot2:
            current_position_2.x = observations[1][0][0]
            current_position_2.y = observations[1][0][1]
            current_position_2.z = 0.0

            # for marobot3:
            current_position_3.x = observations[2][0][0]
            current_position_3.y = observations[2][0][1]
            current_position_3.z = 0.0

            MAX_X = 16.0
            MIN_X = -16.0
            MAX_Y = 16.0
            MIN_Y = -16.0

            # We see if we are outside the Learning Space or get into desired points
            # difine type dictionary in order to decide if go to the desired point
            # print("current_position_1 is:", current_position_1)
            num = 0
            desired_current_position = {str(current_position_1): self.marobot1_desired_point, str(current_position_2): self.marobot2_desired_point, str(current_position_3): self.marobot3_desired_point}
            obstacle_current_position = {str(current_position_1): self.marobot1_obstacle_point, str(current_position_2): self.marobot2_obstacle_point, str(current_position_3): self.marobot3_obstacle_point}
            for current_position in [current_position_1, current_position_2, current_position_3]:
                if self._episode_dones[num] is False:
                    if current_position.x <= MAX_X and current_position.x > MIN_X:
                        if current_position.y <= MAX_Y and current_position.y > MIN_Y:
                            rospy.logdebug(
                                "TurtleBot Position is OK ==>[" + str(current_position.x) + "," + str(current_position.y) + "]")

                            # We see if it got to the desired point
                            if self.is_in_desired_position(desired_current_position[str(current_position)], current_position):
                                self._episode_done = True
                            # else:
                            #     self._episode_done = False
                            elif self.is_in_obstacle_position(obstacle_current_position[str(current_position)], current_position):
                                self._episode_done = True
                            else:
                                self._episode_done = False

                        else:
                            rospy.logerr("TurtleBot to Far in Y Pos ==>" + str(current_position.x))
                            self._episode_done = True
                    else:
                        rospy.logerr("TurtleBot to Far in X Pos ==>" + str(current_position.x))
                        self._episode_done = True
                    print("Agent num is:", num)
                    print("goal_Env_done is:", self._episode_done)
                # sub_episode_done = sub_episode_done.append(self._episode_done)
                #     sub_episode_done.append(self._episode_done)
                    self._episode_dones[num] = self._episode_done
                else:
                    self._episode_dones[num] = True

                num = num +1

            # self._episode_dones = sub_episode_done[:]
            print("all robot dones are", self._episode_dones)

            #add
            # self._episode_dones[1] = True
            # self._episode_dones[2] = True

            return self._episode_dones


    # define reward for all robots through distance between each robot and desired point or has crashed into each other

    def _compute_reward(self, observations, dones):
        # define and store all reward for different robots
        reward_all = [0,0,0]

        current_position_1 = Point()
        current_position_2 = Point()
        current_position_3 = Point()

        # for marobot1:
        current_position_1.x = observations[0][0][0]
        current_position_1.y = observations[0][0][1]
        current_position_1.z = 0.0
        laser_data_1 = observations[0][1]

        # for marobot2:
        current_position_2.x = observations[1][0][0]
        current_position_2.y = observations[1][0][1]
        current_position_2.z = 0.0
        laser_data_2 = observations[1][1]

        # for marobot3:
        current_position_3.x = observations[2][0][0]
        current_position_3.y = observations[2][0][1]
        current_position_3.z = 0.0
        laser_data_3 = observations[2][1]


        #obtain all robots given to the desired points
        #Agents are rewarded based on minimum agent distance to each desired point, penalized for collisions
        #establish reward for each robot and there are three conditions: each distance to desired point, all reached desired point and each crashed
        distance_from_des_points = []
        distance_differences = []
        # distance_from_start = [3,3,3]
        i = -1
        for current_position in [current_position_1, current_position_2, current_position_3]:
            i += 1
            if i == 0:
                distance_from_des_point = self.get_distance_from_desired_point_1(current_position)
            elif i == 1:
                distance_from_des_point = self.get_distance_from_desired_point_2(current_position)
            elif i == 2:
                distance_from_des_point = self.get_distance_from_desired_point_3(current_position)
            distance_difference = distance_from_des_point - self.previous_distance_from_des_points[i]
            # distance_difference = (distance_from_des_point - distance_from_start[i])/100.00
            distance_from_des_points.append(distance_from_des_point)
            distance_differences.append(distance_difference)

        self.previous_distance_from_des_points = distance_from_des_points[:]
        # distance_difference = distance_from_des_point - self.previous_distance_from_des_point



        #------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>
        #print("First time reward_all is:", reward_all)
        # ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>
        # original code:
        # if not done:
        #     if self.last_action == "FORWARDS":
        #         reward = -1*self.forwards_reward
        #     else:
        #         reward = self.turn_reward
        #     # else:
        #     # reward = -1*self.end_episode_points
        #
        #     if distance_difference < 0.0:
        #         rospy.logwarn("DECREASE IN DISTANCE GOOD")
        #         reward += self.forwards_reward
        #         # reward = 100
        #     else:
        #         rospy.logerr("ENCREASE IN DISTANCE BAD")
        #         # reward += 0
        #         reward = reward - 10*distance_difference
        # else:
        #     if self.is_in_desired_position(current_position):
        #         reward = self.end_episode_points
        #     else:
        #         reward = -1 * self.end_episode_points


        # Situation2: all robots reach desired point, currently adopt independt network, so we don't need all reward
        is_in_desired_positions = [self.is_in_desired_position(self.marobot1_desired_point, current_position_1),
                                   self.is_in_desired_position(self.marobot2_desired_point, current_position_2),
                                   self.is_in_desired_position(self.marobot3_desired_point, current_position_3)]
        is_in_desired_position_total = is_in_desired_positions[0] and is_in_desired_positions[1] and \
                                       is_in_desired_positions[2]
        # obstacle_point, current_position
        is_in_obstacle_positions = [self.is_in_obstacle_position(self.marobot1_obstacle_point, current_position_1),
                                    self.is_in_obstacle_position(self.marobot1_obstacle_point, current_position_2) or self.is_in_obstacle_position(self.marobot3_obstacle_point, current_position_2) ,
                                    self.is_in_obstacle_position(self.marobot3_obstacle_point, current_position_3)]

        has_crashed_all = [self.has_crashed(self.min_laser_value, laser_data_1, '1'),
                           self.has_crashed(self.min_laser_value, laser_data_2, '2'),
                           self.has_crashed(self.min_laser_value, laser_data_3, '3')]


        # if is_in_desired_position_total:
        # reward_all = [reward+10 for reward in reward_all]
        # if is_in_desired_position_total:
        #     reward_all = [reward+20 for reward in reward_all]

        # Each agent is rewarded when each agent reaches to the desired points
        # case3:
        # for desired_position in is_in_desired_positions:
        #
        #     if desired_position == True:
        #         # reward_all[m] += 5
        #         reward_all[m] += 500

        # Agents are rewarded based on minimum agent distance to each desired point
        #dists = min(distance_from_des_points)
        #obtain reward for each robot and store in reward_all
        #Situation1: define each reward for each robot
        # reward -= dists
        # transfer different data type
        # reward as the distance difference is better than the direct distance
        # distance_from_des_points_np = numpy.array(distance_from_des_points)

        for i, reward_each in enumerate(reward_all):

            if dones[i] is False:

                # new add in situation: there is no move, otherwise push each agent
                # case1:
                # n = -1
                # for distance in distance_differences:
                #     n += 1
                #     if distance == 0:
                #         reward_all[n] -= 1


                # case2:
                # distance_differences_np = numpy.array(distance_differences)
                # reward_all_np = numpy.array(reward_all)
                # reward_all_np = reward_all_np - distance_differences_np * 10
                # reward_all = reward_all_np.tolist()

                ##-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>
                reward_each = reward_each - distance_differences[i]*10
                reward_all[i] = reward_each
                ##-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>


                # ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>
                # print("Second time reward_all is:", reward_all)
                # ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>


                #case3-modified:
                if is_in_desired_positions[i] is True:
                    reward_all[i] += 200

                # ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>
                # print("Third time reward_all is:", reward_all)
                # ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>

                # Agents are penalty based on laser data for each robot
                # case4:
                # has_crashed_all = [self.has_crashed(self.min_laser_value,laser_data_1,'1'),self.has_crashed(self.min_laser_value,laser_data_2,'2'),self.has_crashed(self.min_laser_value,laser_data_3,'3')]
                # j = -1
                # for crashed in has_crashed_all:
                #     j += 1
                #     if crashed == True:
                #         # reward_all[j] -= 10
                #         reward_all[j] -= 2


                # case4 - modified:
                # if has_crashed_all[i] is True or is_in_obstacle_positions[i]:
                #     reward_all[i] -= 5
                if has_crashed_all[i] is True:
                    reward_all[i] -= 0

                if is_in_obstacle_positions[i] is True:
                    reward_all[i] -= 0

                # ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>
                # print("Forth time reward_all is:", reward_all)
                # ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>

            elif dones[i] is True and self._if_dones_label[i] is False:
                #case2:
                # distance_differences_np = numpy.array(distance_differences)
                # reward_all_np = numpy.array(reward_all)
                # reward_all_np = reward_all_np - distance_differences_np*10
                # reward_all = reward_all_np.tolist()

                #------------------------------------------------------
                reward_each = reward_each - distance_differences[i]*10
                reward_all[i] = reward_each
                # ------------------------------------------------------

                #------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>
                # print("Second time reward_all is:", reward_all)
                #------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>

                #Situation2: all robots reach desired point, currently adopt independt network, so we don't need all reward
                # is_in_desired_positions = [self.is_in_desired_position(self.marobot1_desired_point,current_position_1),self.is_in_desired_position(self.marobot2_desired_point,current_position_2),self.is_in_desired_position(self.marobot3_desired_point,current_position_3)]
                # is_in_desired_position_total = is_in_desired_positions[0] and is_in_desired_positions[1] and is_in_desired_positions[2]
                # if is_in_desired_position_total:
                # reward_all = [reward+10 for reward in reward_all]
                # if is_in_desired_position_total:
                #     reward_all = [reward+20 for reward in reward_all]


                #Each agent is rewarded when each agent reaches to the desired points
                #case3-modified:
                # m = -1
                # for desired_position in is_in_desired_positions:
                #     m += 1
                #     if desired_position == True:
                #         # reward_all[m] += 5
                #         reward_all[m] += 500
                if is_in_desired_positions[i] is True:
                    reward_all[i] += 200

                # ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>
                # print("Third time reward_all is:", reward_all)
                # ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>

                # Agents are penalty based on laser data for each robot
                #case4:
                # has_crashed_all = [self.has_crashed(self.min_laser_value,laser_data_1,'1'),self.has_crashed(self.min_laser_value,laser_data_2,'2'),self.has_crashed(self.min_laser_value,laser_data_3,'3')]
                # j = -1
                # for crashed in has_crashed_all:
                #     j += 1
                #     if crashed == True:
                #         # reward_all[j] -= 10
                #         reward_all[j] -= 2

                # case4 - modified:
                if has_crashed_all[i] is True :
                    reward_all[i] -= 0

                if  is_in_obstacle_positions[i] is True:
                    reward_all[i] -= 0
                # ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>
                #print("Forth time reward_all is:", reward_all)
                # ------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>



                # rospy.logdebug("reward=" + str(reward))
                # self.cumulated_reward += reward
                # rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
                # self.cumulated_steps += 1
                # rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
                self._if_dones_label[i] = True
            else:
                reward_all[i] = 0

        
        return reward_all


    # Internal TaskEnv Methods
    
    def discretize_observation(self,data,new_ranges,pub_num):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False
        
        discretized_ranges = []
        filtered_range = []
        mod = len(data.ranges)/new_ranges # In the case of real robots
        # mod = new_ranges # In the term of simulation
        
        max_laser_value = data.range_max
        min_laser_value = data.range_min
        
        rospy.logdebug("data=" + str(data))
        rospy.logwarn("mod=" + str(mod))
        
        for i, item in enumerate(data.ranges):
            # if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    #discretized_ranges.append(self.max_laser_value)
                    discretized_ranges.append(round(max_laser_value,self.dec_obs))
                elif numpy.isnan(item):
                    #discretized_ranges.append(self.min_laser_value)
                    discretized_ranges.append(round(min_laser_value,self.dec_obs))
                else:
                    #discretized_ranges.append(int(item))
                    discretized_ranges.append(round(item,self.dec_obs))
                    
                if (self.min_range > round(item,self.dec_obs) > 0):
                    rospy.logerr("Agent number is"+pub_num+"and"+"done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    # self._episode_done = True
                    self._episode_dones[int(pub_num)-1] = True
                    #__________________________________________________________>>>>>>>>>>>>>>>>
                    print("crash robot number is", pub_num)
                    print("whether crshed or not is", self._episode_dones[int(pub_num)-1])
                    # __________________________________________________________>>>>>>>>>>>>>>>>
                else:
                    # rospy.logwarn("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    pass
                # We add last value appended
                filtered_range.append(discretized_ranges[-1])
            # else:
                # We add value zero
                # filtered_range.append(0.1)
                    
        rospy.logdebug("Size of observations, discretized_ranges==>"+str(len(discretized_ranges)))
        
        
        self.publish_filtered_laser_scan(   laser_original_data=data,
                                            new_filtered_laser_range=discretized_ranges,
                                            pub_num=pub_num)
        
        return discretized_ranges
        
    
    def publish_filtered_laser_scan(self, laser_original_data, new_filtered_laser_range, pub_num):
        
        rospy.logdebug("new_filtered_laser_range==>"+str(new_filtered_laser_range))
        
        laser_filtered_object = LaserScan()

        h = Header()
        h.stamp = rospy.Time.now() # Note you need to call rospy.init_node() before this will work
        h.frame_id = laser_original_data.header.frame_id
        
        laser_filtered_object.header = h
        laser_filtered_object.angle_min = laser_original_data.angle_min
        laser_filtered_object.angle_max = laser_original_data.angle_max
        
        new_angle_incr = abs(laser_original_data.angle_max - laser_original_data.angle_min) / len(new_filtered_laser_range)
        
        #laser_filtered_object.angle_increment = laser_original_data.angle_increment
        laser_filtered_object.angle_increment = new_angle_incr
        laser_filtered_object.time_increment = laser_original_data.time_increment
        laser_filtered_object.scan_time = laser_original_data.scan_time
        laser_filtered_object.range_min = laser_original_data.range_min
        laser_filtered_object.range_max = laser_original_data.range_max
        
        laser_filtered_object.ranges = []
        laser_filtered_object.intensities = []
        for item in new_filtered_laser_range:
            # if item == 0.0:
            #     # laser_distance = 0.1
            #     laser_distance = 0.0
            # else:
            laser_distance = item
            laser_filtered_object.ranges.append(laser_distance)
            laser_filtered_object.intensities.append(item)
        
        if pub_num == '1':
            self.laser_filtered_pub_1.publish(laser_filtered_object)
        elif pub_num == '2':
            self.laser_filtered_pub_2.publish(laser_filtered_object)
        elif pub_num == '3':
            self.laser_filtered_pub_3.publish(laser_filtered_object)



    def get_distance_from_desired_point_1(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.marobot1_desired_point)

        return distance

    def get_distance_from_desired_point_2(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.marobot2_desired_point)

        return distance

    def get_distance_from_desired_point_3(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.marobot3_desired_point)

        return distance

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((float(pstart.x), float(pstart.y), float(pstart.z)))
        b = numpy.array((float(p_end.x), float(p_end.y), float(p_end.z)))

        distance = numpy.linalg.norm(a - b)

        return distance

    def is_in_desired_position(self, desired_point, current_position, epsilon=0.2):
        """
        It return True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False

        x_pos_plus = desired_point.x + epsilon
        x_pos_minus = desired_point.x - epsilon
        y_pos_plus = desired_point.y + epsilon
        y_pos_minus = desired_point.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        return is_in_desired_pos


    def is_in_obstacle_position(self, obstacle_point, current_position, inflation=0.3):
        """
        It return True if the current position is similar to the obstacle poistion
        """

        is_in_obstacle_pos = False

        x_pos_plus = obstacle_point.x + inflation
        x_pos_minus = obstacle_point.x - inflation
        y_pos_plus = obstacle_point.y + inflation
        y_pos_minus = obstacle_point.y - inflation

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_obstacle_pos = x_pos_are_close and y_pos_are_close
        print('is_in_obstacle_pos is :>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',is_in_obstacle_pos)
        return is_in_obstacle_pos

