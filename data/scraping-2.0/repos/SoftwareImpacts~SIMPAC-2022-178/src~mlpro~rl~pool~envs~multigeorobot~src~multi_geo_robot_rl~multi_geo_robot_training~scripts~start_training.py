#!/usr/bin/env python

import gym
import numpy
import time
from gym import wrappers

# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

from stable_baselines3 import PPO


if __name__ == '__main__':

    rospy.init_node('multi_geo_robot_learning', anonymous=True, log_level=rospy.DEBUG)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/multi_geo_robot/reaching_task/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name,20)

    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('rl_training')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    rospy.sleep(5.0)

    obs = env.reset()

    model = PPO("MlpPolicy", env, n_steps=5, verbose=1)
    model.learn(total_timesteps=10000)

    # while 1:
    #     try:
    #         rospy.sleep(1.0)
    #     except KeyboardInterrupt:
    #         env.close()
    #         break

    env.close()