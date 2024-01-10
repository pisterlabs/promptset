from collections import OrderedDict

import numpy as np
from gym import GoalEnv, spaces

#!/usr/bin/env python

# IMPORT
from numpy.core.defchararray import upper
import gym
import rospy
import numpy as np
import time
import random
import sys
import yaml
import math
import datetime
import rospkg
from gym import utils, spaces
from gym.utils import seeding
from gym.envs.registration import register
#from tf.transformations import quaternion_from_euler
from scipy.spatial.transform import Rotation
from collections import OrderedDict

# OTHER FILES
import util_env as U
import math_util as UMath
#from environments.gazebo_connection import GazeboConnection
#from environments.controllers_connection import ControllersConnection
#from environments.joint_publisher import JointPub
from joint_array_publisher import JointArrayPub
#from baselines import logger
import logger
# MESSAGES/SERVICES
from std_msgs.msg import String
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
#from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import Image
#from gazebo_msgs.srv import GetModelState
#from gazebo_msgs.srv import SetModelState
#from gazebo_msgs.srv import GetLinkState
from geometry_msgs.msg import Point, Quaternion, Vector3
#from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point, PoseStamped
from openai_ros.msg import RLExperimentInfo
from moveit_msgs.msg import MoveGroupActionFeedback

class AuboRobotFetch(GoalEnv):
    """
    Simple bit flipping env, useful to test HER.
    The goal is to flip all the bits to get a vector of ones.
    In the continuous variant, if the ith action component has a value > 0,
    then the ith bit will be flipped.
    :param n_bits: (int) Number of bits to flip
    :param continuous: (bool) Whether to use the continuous actions version or not,
        by default, it uses the discrete one
    :param max_steps: (int) Max number of steps, by default, equal to n_bits
    :param discrete_obs_space: (bool) Whether to use the discrete observation
        version or not, by default, it uses the MultiBinary one
    """
    def __init__(self, joint_increment=None, sim_time_factor=0.005, random_object=False, random_position=False,
                 use_object_type=False, populate_object=False, env_object_type='free_shapes'):
        super(AuboRobotFetch, self).__init__()
        # The achieved goal is determined by the current state
        # here, it is a special where they are equal
        
            # In the discrete case, the agent act on the binary
            # representation of the observation
        self.observation_space = spaces.Dict({
            'observation': spaces.Discrete(6),
            'achieved_goal': spaces.Discrete(6),
            'desired_goal': spaces.Discrete(6)
        })
       
        if self._joint_increment is None:
            low_action = np.array([
                -(math.pi - 0.05),
                -(math.pi - 0.05),
                -(math.pi - 0.05),
                -(math.pi - 0.05),
                -(math.pi - 0.05),
                -(math.pi - 0.05)])

            high_action = np.array([
                math.pi - 0.05,
                math.pi - 0.05,
                math.pi - 0.05,
                math.pi - 0.05,
                math.pi - 0.05,
                math.pi - 0.05])
        else: # Use joint_increments as action
            low_action = np.array([
                -self._joint_increment,
                -self._joint_increment,
                -self._joint_increment,
                -self._joint_increment,
                -self._joint_increment,
                -self._joint_increment])

            high_action = np.array([
                self._joint_increment,
                self._joint_increment,
                self._joint_increment,
                self._joint_increment,
                self._joint_increment,
                self._joint_increment])
        self.action_space = gym.spaces.Box(low_action, high_action)

        high = np.array([
            999,
            math.pi,
            math.pi,
            math.pi,
            math.pi,
            math.pi,
            math.pi,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            1,
            1.4,
            1.5,
            999])

        low = np.array([
            0,
            -math.pi,
            -math.pi,
            -math.pi,
            -math.pi,
            -math.pi,
            -math.pi,
            0,
            0,
            -1,
            0,
            0,
            0])

        if self._use_object_type:
            high = np.append(high, 9)
            low = np.append(low, 0)
        
        self.obs_space = gym.spaces.Box(low, high)
        
        
        self.state = None
        self.desired_goal = np.array([-1.58, 1.2, -1.4, -1.87, 1.5, 0])
        max_steps = 100
        self.max_steps = max_steps
        self.current_step = 0
        self.reset()

    def convert_if_needed(self, state):
        """
        Convert to discrete space if needed.
        :param state: (np.ndarray)
        :return: (np.ndarray or int)
        """
        if self.discrete_obs_space:
            # The internal state is the binary representation of the
            # observed one
            return int(sum([state[i] * 2**i for i in range(len(state))]))
        return state

    def _get_obs(self):
        """
        Helper to create the observation.
        :return: (OrderedDict<int or ndarray>)
        """
        return OrderedDict([
            ('observation', self.convert_if_needed(self.state.copy())),
            ('achieved_goal', self.convert_if_needed(self.state.copy())),
            ('desired_goal', self.convert_if_needed(self.desired_goal.copy()))
        ])

    def reset(self):
        self.current_step = 0
        self.state = self.obs_space.sample()
        return self._get_obs()

    def step(self, action):
        if self.continuous:
            self.state[action > 0] = 1 - self.state[action > 0]
        else:
            self.state[action] = 1 - self.state[action]
        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], None)
        done = reward == 0
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        info = {'is_success': done}
        done = done or self.current_step >= self.max_steps
        return obs, reward, done, info

    def compute_reward(self,
                       achieved_goal: np.ndarray,
                       desired_goal: np.ndarray,
                       _info) -> float:
        # Deceptive reward: it is positive only when the goal is achieved
        if self.discrete_obs_space:
            return 0.0 if achieved_goal == desired_goal else -1.0
        return 0.0 if (achieved_goal == desired_goal).all() else -1.0

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.state.copy()
        print(self.state)

    def close(self):
        pass