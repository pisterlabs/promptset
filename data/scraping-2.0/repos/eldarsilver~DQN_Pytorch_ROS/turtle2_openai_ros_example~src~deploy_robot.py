#!/usr/bin/env python

import rospy
import numpy 
import time
import math
from gym import spaces
#from openai_ros.robot_envs import turtlebot2_env
#from gym.envs.registration import register
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
#from openai_ros.openai_ros_common import ROSLauncher
import os

from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime

from std_msgs.msg import String
#from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import torch
import torch.nn as nn

import pickle



class DQN(nn.Module):
    # hidden_size=64
    def __init__(self, inputs, outputs, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=inputs, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=outputs)
        #self.fc5 = nn.Linear(in_features=16, out_features=outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc4(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        #x = self.fc5(x)
        return x




class rlComponent(object):

    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot2 in some kind of maze.
        It will learn how to move around the maze without crashing.
        """

        # Only variable needed to be set here
        number_actions = rospy.get_param('~n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        #number_observations = rospy.get_param('/turtlebot2/n_observations')


        # Actions and Observations
        self.dec_obs = rospy.get_param(
            "~number_decimals_precision_obs", 1)
        self.linear_forward_speed = rospy.get_param(
            '~linear_forward_speed')
        self.linear_turn_speed = rospy.get_param(
            '~linear_turn_speed')
        self.angular_speed = rospy.get_param('~angular_speed')
        self.init_linear_forward_speed = rospy.get_param(
            '~init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param(
            '~init_linear_turn_speed')

        self.n_observations = rospy.get_param('~n_observations')
        self.min_range = rospy.get_param('~min_range')
        self.max_laser_value = rospy.get_param('~max_laser_value')
        self.min_laser_value = rospy.get_param('~min_laser_value')

        MODEL_CKPT = rospy.get_param('~model_ckpt')


        self.actions = range(number_actions)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.policy = DQN(self.n_observations, number_actions).to(self.device)
        self.policy.load_state_dict(torch.load(MODEL_CKPT, map_location=self.device))
        self.policy.eval()


        self._cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)
        self.last_action = "FORWARDS"
      
        self.laser_scan = None
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)
        laser_scan = self._check_laser_scan_ready()
        rospy.logdebug("laser_scan len===>"+str(len(laser_scan.ranges)))


        # Number of laser reading jumped
        self.new_ranges = int(
            math.ceil(float(len(laser_scan.ranges)) / float(self.n_observations)))

        rospy.logdebug("n_observations===>"+str(self.n_observations))
        rospy.logdebug(
            "new_ranges, jumping laser readings===>"+str(self.new_ranges))

        high = numpy.full((self.n_observations), self.max_laser_value)
        low = numpy.full((self.n_observations), self.min_laser_value)

        # We only use two integers
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" +
                       str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("~forwards_reward")
        self.turn_reward = rospy.get_param("~turn_reward")
        self.end_episode_points = rospy.get_param(
            "~end_episode_points")

        self.cumulated_steps = 0.0

        self.laser_filtered_pub = rospy.Publisher(
            '/scan_filtered', LaserScan, queue_size=1)
        self._init_env_variables()
        self._set_init_pose()
        rospy.spin()





    def _laser_scan_callback(self, data):
        self.laser_scan = data




    def get_laser_scan(self):
        return self.laser_scan



    def _check_laser_scan_ready(self):
        #self.laser_scan = None
        rospy.logdebug("Waiting for /scan to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message("/scan", LaserScan, timeout=5.0)
                rospy.logdebug("Current /scan READY=>")

            except:
                rospy.logerr("Current /scan not ready yet, retrying for getting laser_scan")
        return self.laser_scan





    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
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
        #laser_scan = self.get_laser_scan()
        discretized_ranges = self.laser_scan.ranges
        self.publish_filtered_laser_scan(laser_original_data=self.laser_scan,
                                         new_filtered_laser_range=discretized_ranges)

        self.step()





    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()

        rospy.logdebug("BEFORE DISCRET _episode_done==>" +
                       str(self._episode_done))

        discretized_observations = self.discretize_observation(laser_scan,
                                                               self.new_ranges
                                                               )

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("AFTER DISCRET_episode_done==>"+str(self._episode_done))
        rospy.logdebug("END Get Observation ==>")
        return discretized_observations




    def _is_done(self, observations):

        if self._episode_done:
            rospy.logdebug("TurtleBot2 is Too Close to wall" +
                           str(self._episode_done))
        else:
            rospy.logerr("TurtleBot2 is Ok")

        return self._episode_done




    def _compute_reward(self, observations, done):

        if not done:
            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward
        else:
            reward = -1*self.end_episode_points

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    # Internal TaskEnv Methods




    def discretize_observation(self, data, new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        filtered_range = []
        #mod = len(data.ranges)/new_ranges
        mod = new_ranges

        max_laser_value = data.range_max
        min_laser_value = data.range_min

        rospy.logdebug("data=" + str(data))
        rospy.logwarn("data.range_max= %s" % data.range_max)
        rospy.logwarn("data.range_min= %s" % data.range_min)
        rospy.logwarn("len(data.ranges)= %s" % len(data.ranges))
        rospy.logwarn("data.angle_min)= %s" % data.angle_min)
        rospy.logwarn("data.angle_max)= %s" % data.angle_max)
        rospy.logwarn("data.angle_increment= %s" % data.angle_increment)
        rospy.logwarn("mod=" + str(mod))


        rospy.loginfo('right data.ranges[89] %s' % data.ranges[89])
        rospy.loginfo('left data.ranges[269] %s ' % data.ranges[269]) 
        rospy.loginfo('back data.ranges[359] %s' % data.ranges[359])
        rospy.loginfo('back data.ranges[0] %s' % data.ranges[0])
        rospy.loginfo('front data.ranges[179] %s' % data.ranges[179])

        
        idx_ranges = [89, 135, 179, 224, 269]

        for item in idx_ranges:
            if data.ranges[item] == float('Inf') or numpy.isinf(data.ranges[item]):
                # discretized_ranges.append(self.max_laser_value)
                discretized_ranges.append(round(max_laser_value, self.dec_obs))
            elif numpy.isnan(data.ranges[item]):
                # discretized_ranges.append(self.min_laser_value)
                discretized_ranges.append(round(min_laser_value, self.dec_obs))
            else:
                # discretized_ranges.append(int(item))
                discretized_ranges.append(round(data.ranges[item], self.dec_obs))

            if (self.min_range > data.ranges[item] > 0):
                rospy.logerr("done Validation >>> data.ranges[item]=" + str(data.ranges[item])+"< "+str(self.min_range))
                self._episode_done = True
            else:
                rospy.logwarn("NOT done Validation >>> data.ranges[item]=" + str(data.ranges[item])+"< "+str(self.min_range))

        #rospy.logdebug("Size of observations, discretized_ranges==>"+str(len(discretized_ranges))) 

        return discretized_ranges       
        """
        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                if item == float('Inf') or numpy.isinf(item):
                    # discretized_ranges.append(self.max_laser_value)
                    discretized_ranges.append(
                        round(max_laser_value, self.dec_obs))
                elif numpy.isnan(item):
                    # discretized_ranges.append(self.min_laser_value)
                    discretized_ranges.append(
                        round(min_laser_value, self.dec_obs))
                else:
                    # discretized_ranges.append(int(item))
                    discretized_ranges.append(round(item, self.dec_obs))

                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" +
                                 str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logwarn("NOT done Validation >>> item=" +
                                  str(item)+"< "+str(self.min_range))
                # We add last value appended
                filtered_range.append(discretized_ranges[-1])
            else:
                # We add value zero
                filtered_range.append(0.1)

        rospy.logdebug(
            "Size of observations, discretized_ranges==>"+str(len(discretized_ranges)))

        self.publish_filtered_laser_scan(laser_original_data=data,
                                         new_filtered_laser_range=discretized_ranges)
        
        return discretized_ranges
        """



    def publish_filtered_laser_scan(self, laser_original_data, new_filtered_laser_range):

        rospy.logdebug("new_filtered_laser_range==>" +
                       str(new_filtered_laser_range))

        laser_filtered_object = LaserScan()

        h = Header()
        # Note you need to call rospy.init_node() before this will work
        h.stamp = rospy.Time.now()
        h.frame_id = laser_original_data.header.frame_id

        laser_filtered_object.header = h
        laser_filtered_object.angle_min = laser_original_data.angle_min
        laser_filtered_object.angle_max = laser_original_data.angle_max

        new_angle_incr = abs(laser_original_data.angle_max -
                             laser_original_data.angle_min) / len(new_filtered_laser_range)

        #laser_filtered_object.angle_increment = laser_original_data.angle_increment
        laser_filtered_object.angle_increment = new_angle_incr
        laser_filtered_object.time_increment = laser_original_data.time_increment
        laser_filtered_object.scan_time = laser_original_data.scan_time
        laser_filtered_object.range_min = laser_original_data.range_min
        laser_filtered_object.range_max = laser_original_data.range_max

        laser_filtered_object.ranges = []
        laser_filtered_object.intensities = []
        for item in new_filtered_laser_range:
            if item == 0.0:
                laser_distance = 0.1
            else:
                laser_distance = item
            laser_filtered_object.ranges.append(laser_distance)
            laser_filtered_object.intensities.append(item)

        self.laser_filtered_pub.publish(laser_filtered_object)





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
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logwarn("Move Base")
        rospy.logwarn("linear_speed %d", linear_speed)
        rospy.logwarn("angular_speed %d", angular_speed)
        #rospy.logdebug("TurtleBot2 Base Twist Cmd>>" + str(cmd_vel_value))
        #self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        time.sleep(0.2)
        #time.sleep(0.02)
        """
        self.wait_until_twist_achieved(cmd_vel_value,
                                        epsilon,
                                        update_rate,
                                        min_laser_distance)
        """




    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action %d", action)
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0:  # FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
            rospy.logwarn("Action 0 F")
        elif action == 1:  # LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
            rospy.logwarn("Action 1 L")
        elif action == 2:  # RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_RIGHT"
            rospy.logwarn("Action 2 R")                           
        elif self._episode_done == True: # Stop
            linear_speed = 0.0
            angular_speed = 0.0
            self.last_action = "STOP"
            rospy.logwarn("Action end")

        # We tell TurtleBot2 the linear and angular speed to set to execute
        
        self.move_base(linear_speed,
                       angular_speed,
                       epsilon=0.05,
                       update_rate=10,
                       min_laser_distance=self.min_range)
        
        #rospy.logdebug("END Set Action ==>"+str(action) +", NAME="+str(self.last_action))
        rospy.logwarn("END Set Action %d", action)



    
    def select_action(self, policy, state):
        #rospy.logwarn("state.shape: ")
        #rospy.logwarn(state.shape)
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policy.eval()
            action = policy(state).max(axis=1)[1].view(1, 1)
        return action




    def step(self):
        obs = self._get_obs()
        obs = [round(num, 1) for num in obs]
        rospy.loginfo("obs %s" % obs)
        while obs != [] and self._episode_done == False:
            state = torch.from_numpy(numpy.array(obs)).float().unsqueeze(0).to(self.device)
            rospy.loginfo('state %s' % state)
            # Pick an action based on the current state
            action_dq = self.select_action(self.policy, state)
            rospy.logwarn("Next actionq is:%d", action_dq)
            # Execute the action in the environment and get feedback
            #self._set_action(action_dq)
            rospy.logwarn("Start Set Action %d", action_dq)
         
            if action_dq == 0:  # FORWARD
                linear_speed = self.linear_forward_speed
                angular_speed = 0.0
                self.last_action = "FORWARDS"
                rospy.logwarn("linear_speed %d", linear_speed)
                rospy.logwarn("angular_speed %d", angular_speed)
            elif action_dq == 1:  # LEFT
                linear_speed = self.linear_turn_speed
                angular_speed = self.angular_speed
                self.last_action = "TURN_LEFT"
                rospy.logwarn("linear_speed %d", linear_speed)
                rospy.logwarn("angular_speed %d", angular_speed)
            elif action_dq == 2:  # RIGHT
                linear_speed = self.linear_turn_speed
                angular_speed = -1*self.angular_speed
                self.last_action = "TURN_RIGHT"
                rospy.logwarn("linear_speed %d", linear_speed)
                rospy.logwarn("angular_speed %d", angular_speed)
            elif self._episode_done == True: # Stop
                linear_speed = 0.0
                angular_speed = 0.0
                self.last_action = "STOP"
                rospy.logwarn("linear_speed %d", linear_speed)
                rospy.logwarn("angular_speed %d", angular_speed)
            # We tell TurtleBot2 the linear and angular speed to set to execute
        
            self.move_base(linear_speed,
                           angular_speed,
                           epsilon=0.05,
                           update_rate=10,
                           min_laser_distance=self.min_range)
        
            
            rospy.logwarn("END Set Action %d", action_dq)

            obs = self._get_obs()
            obs = [round(num, 1) for num in obs]
             

       

if __name__ == '__main__':
    try:
        rospy.init_node('re_fr', anonymous=False)
        rlComp = rlComponent()
        #while rlComp.ok():
        #	pass
    except rospy.ROSInterruptException:
        pass


