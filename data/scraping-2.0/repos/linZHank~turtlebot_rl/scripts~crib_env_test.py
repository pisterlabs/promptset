#!/usr/bin/env python
from __future__ import print_function

import gym
import numpy as np
import time
import random
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates, LinkStates

# import our training environment
from openai_ros_envs import crib_task_env # need write task env

rospy.init_node('env_test', anonymous=True, log_level=rospy.DEBUG)    
env = gym.make('TurtlebotCrib-v0')


for episode in range(100000):
  state, info = env.reset()
  done = False
  for step in range(128):
    action = random.randrange(4)
    next_state, reward, done, info = env.step(action)
    print("Episode : {}, Step: {}, \nCurrent position: {}, Goal position: {}, Reward: {:.4f}".format(
      episode,
      step,
      info["current_position"],
      info["goal_position"],
      reward
    ))
    if done:
      break
    
