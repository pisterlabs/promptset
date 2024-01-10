from gym import spaces
from openai_ros.robot_envs import hallway_env
from gym.envs.registration import register
from geometry_msgs.msg import Twist
import cv2
import ros_numpy

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import rospy
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from math import exp
import time
import math

# The path is __init__.py of openai_ros, where we import the MovingCubeOneDiskWalkEnv directly
timestep_limit_per_episode = 1000  # Can be any Value

register(
    id='HallwayCollisionLaser-v0',
    entry_point='openai_ros:task_envs.hallway.hallway_collision_new.HallwayCollisionAvoidance',
    timestep_limit=timestep_limit_per_episode,
)


class HallwayCollisionAvoidance(hallway_env.HallwayEnv):
    def __init__(self):

        self.action_space = spaces.Discrete(3)
        self.reward_range = (-np.inf, np.inf)

        # Observation space : 100 laser scan ranges + 2 odometry
        self.n_observations = 100
        self.max_laser_value = 20.0
        self.min_laser_value = 0.02
        high = np.full((self.n_observations), self.max_laser_value)
        low = np.full((self.n_observations), self.min_laser_value)

        # add odometery to observations
        self.n_observations += 3
        self.high = np.append(high, [10, 10, 1])
        self.low = np.append(low, [-10, -10, -1])

        # laser range as observation space
        self.observation_space = spaces.Box(low=self.low, high=self.high,dtype=np.float32)

        # kinect depth as observation space
        # self.observation_space = spaces.Box(low=0, high=255, shape=(64, 128, 3), dtype=np.float32)

        self.cumulated_steps = 0.0

        # define rewards
        self.move_towards_target = 10
        self.forward_reward = 50
        self.turn_reward = -20
        # self.backward_reward = 2
        # self.stop_reward = -10
        self.end_episode_points = -100

        self.target = 20.0
        self.reward_min_dist_to_goal = 5.0
        self.reward_max_dist_to_goal = -5.0
        self.reward_crashing = -10.0
        self.reward_time_taken = -0.5
        self.reward_goal_reached = 10.0

        self.dec_obs = 5
        self.min_range = 0.35

        self.success = False  # did both the robots cross over safely ?
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(HallwayCollisionAvoidance, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.init_linear_forward_speed = 0
        self.init_linear_turn_speed = 0
        self.move_base(self.init_linear_forward_speed,
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
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)

        # TODO: Add reset of published filtered laser readings
        tom_laser_scan = self.get_tom_laser_scan()
        jerry_laser_scan = self.get_jerry_laser_scan()
        self.last_time = time.time()

        # restart tom
        stop_command = Twist()
        stop_command = Twist()
        stop_command.linear.x = 0.5
        self._tom_cmd_vel_pub.publish(stop_command)

        # wrt the robot's initial position as origin
        self.jerry_last_position = 0.0
        self.tom_last_position = 0.0

        # reset success reward signal
        self.success = False

    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        """
        rospy.logdebug("Start Set Action ==>" + str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0:  # FORWARD
            linear_speed = 0.5
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1:  # LEFT
            linear_speed = 0.1
            angular_speed = 0.5
            self.last_action = "TURN_LEFT"
        elif action == 2:  # RIGHT
            linear_speed = 0.1
            angular_speed = -0.5
            self.last_action = "TURN_RIGHT"
        # elif action == 3: # BACKWARD
        #     linear_speed = -1.0
        #     angular_speed = 0.0
        #     self.last_action = "BACKWARDS"
        # elif action == 3: # STOP
        #     linear_speed = 0.0
        #     angular_speed = 0.0
        #     self.last_action = "STOP"

        # We tell Segbot the linear and angular speed to set to execute
        self.move_base(linear_speed,
                       angular_speed,
                       epsilon=0.05,
                       update_rate=10,
                       min_laser_distance=self.min_range)

        rospy.logdebug("END Set Action ==>" + str(action) + ", NAME=" + str(self.last_action))

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        jerry_laser_scan = self.get_jerry_laser_scan()
        rospy.logdebug("BEFORE DISCRET _episode_done==>" + str(self._episode_done))

        discretized_observations = self.discretize_observation(jerry_laser_scan)

        rospy.logdebug("Observations==>" + str(discretized_observations))
        rospy.logdebug("AFTER DISCRET_episode_done==>" + str(self._episode_done))
        rospy.logdebug("END Get Observation ==>")
        # print('Discretized observations :: ',discretized_observations)

        jerry_odom = self.get_jerry_odom()
        tom_odom = self.get_tom_odom()
        if jerry_odom.pose.pose.position.x <= -10.0:
            self._episode_done = True
            self.success = True
        if tom_odom.pose.pose.position.x >= 10.0:
            stop_command = Twist()
            stop_command.linear.x = 0
            self._tom_cmd_vel_pub.publish(stop_command)
            # self._episode_done = True
        # return np.expand_dims(discretized_observations,axis=0)
        jerry_current_position = 10.0 - self.get_jerry_odom().pose.pose.position.x  # offset for the origin position in gazebo
        if jerry_current_position < 0:
            self._episode_done = True  # if robot leaves hallway without entering

        # add odometry information to observation
        discretized_observations.append(jerry_odom.pose.pose.position.x)
        discretized_observations.append(jerry_odom.pose.pose.position.y)
        discretized_observations.append(jerry_odom.pose.pose.orientation.z)
        # print('observation : ', discretized_observations)

        # normalize observations - min max scaling
        # discretized_observations = (discretized_observations-self.low)/(self.high-self.low)
        return discretized_observations

        # jerry_kinect_depth =  self._get_jerry_kinect_depth()
        # jerry_kinect_depth = ros_numpy.numpify(jerry_kinect_depth)
        # jerry_kinect_depth = cv2.resize(jerry_kinect_depth, dsize=(128, 64), interpolation=cv2.INTER_CUBIC)
        # jerry_kinect_depth_normalized = cv2.normalize(jerry_kinect_depth,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

        # jerry_kinect_rgb = self._get_jerry_kinect_rgb()
        # jerry_kinect_rgb = ros_numpy.numpify(jerry_kinect_rgb)
        # jerry_kinect_rgb = cv2.resize(jerry_kinect_rgb, dsize=(128, 64), interpolation=cv2.INTER_CUBIC)
        # jerry_kinect_rgb_normalized = cv2.normalize(jerry_kinect_rgb, None, alpha=0, beta=1,
        #                                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #
        # cv2.imshow('jerry depth image', jerry_kinect_rgb_normalized)
        # cv2.waitKey(10)
        #
        # jerry_kinect_depth = np.expand_dims(jerry_kinect_rgb, axis=2)
        # print(jerry_kinect_rgb.shape)
        #
        # return jerry_kinect_rgb

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        if self._episode_done:
            rospy.logdebug("Segbot is Too Close to wall==>" + str(self._episode_done))
        else:
            rospy.logerr("Segbot is Ok ==>")

        return self._episode_done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        # if not done:
        #     if self.last_action == "FORWARDS":
        #         reward = self.forward_reward
        #     elif self.last_action == "TURN_LEFT" or self.last_action == "TURN_RIGHT":
        #         reward = self.turn_reward
        #     # elif self.last_action == "BACKWARDS":
        #     #     reward = self.backward_reward
        #     # elif self.last_action == "STOP":
        #     #     reward = self.stop_reward
        #
        #     # reward for moving towards goal in forward direction
        #     jerry_current_position = 10.0 - self.get_jerry_odom().pose.pose.position.x  # offset for the origin position in gazebo
        #     # tom_current_position = self.get_tom_odom().pose.pose.position.x
        #     distance_since_last_reward = jerry_current_position - self.jerry_last_position
        #     if self.last_action=="FORWARDS" and distance_since_last_reward > 0.1:
        #         rospy.logerr("achieved distance reward!")
        #         reward += self.move_towards_target
        #         # set current position as last position in variable.
        #         self.jerry_last_position = jerry_current_position
        #
        #     # reward for maximizing distance from the wall
        #     # print('exp minimum observation is : ',exp(min(observations)))
        #     if exp(min(observations))<50:
        #         reward += exp(min(observations))
        #
        # else:
        #     if self.success == False:
        #         reward = self.end_episode_points
        #     else :
        #         reward = 100000
        #         rospy.logerr("SUCCESS :: Both Robots Successfully Completed crossover !")

        # new rewards, after meeting on April 8th, 2019
        jerry_current_position = 10.0 - self.get_jerry_odom().pose.pose.position.x
        distance_moved_towards_target = jerry_current_position - self.jerry_last_position
        print('distance moved towards goal', distance_moved_towards_target)

        # if not done:
        #     # distance reward
        #     if distance_moved_towards_target > 0 and self.last_action == "FORWARDS":
        #         reward = self.reward_min_dist_to_goal
        #         self.jerry_last_position = jerry_current_position
        #     elif distance_moved_towards_target > 0 and self.last_action != "FORWARDS":
        #         reward = 0.01*self.reward_min_dist_to_goal
        #     else:
        #         reward = self.reward_max_dist_to_goal
        #
        # else:
        #     if self.success:
        #         reward = self.reward_goal_reached
        #     else:
        #         reward = self.reward_crashing

        # modified sunday 14 apr 2019
        if not done:
            reward = distance_moved_towards_target*self.reward_min_dist_to_goal
            if distance_moved_towards_target>0:
                self.jerry_last_position = jerry_current_position

        else:
            if self.success:
                reward = self.reward_goal_reached
            else:
                reward = self.reward_crashing

        # penalize for time taken
        reward += self.reward_time_taken

        rospy.logwarn("reward=" + str(reward))
        # self.cumulated_reward += reward
        # rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        # self.cumulated_steps += 1
        # rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    # Internal TaskEnv Methods
    def discretize_observation(self, data, new_ranges=1):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        filtered_range = []
        # mod = len(data.ranges)/new_ranges
        mod = new_ranges

        max_laser_value = data.range_max
        min_laser_value = data.range_min

        # rospy.logdebug("data=" + str(data))
        # rospy.logwarn("mod=" + str(mod))

        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                if item == float('Inf') or np.isinf(item) or item > max_laser_value:
                    # discretized_ranges.append(self.max_laser_value)
                    discretized_ranges.append(round(max_laser_value, self.dec_obs))
                elif np.isnan(item):
                    # discretized_ranges.append(self.min_laser_value)
                    discretized_ranges.append(round(min_laser_value, self.dec_obs))
                else:
                    # discretized_ranges.append(int(item))
                    discretized_ranges.append(round(item, self.dec_obs))

                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item) + "< " + str(self.min_range))
                    self._episode_done = True
                else:
                    # rospy.logwarn("NOT done Validation >>> item=" + str(item) + "< " + str(self.min_range))
                    pass
                # We add last value appended
                filtered_range.append(discretized_ranges[-1])
            else:
                # We add value zero
                filtered_range.append(0.1)

        rospy.logdebug("Size of observations, discretized_ranges==>" + str(len(discretized_ranges)))

        return discretized_ranges
