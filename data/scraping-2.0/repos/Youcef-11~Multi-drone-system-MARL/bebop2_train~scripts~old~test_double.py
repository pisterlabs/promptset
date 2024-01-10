#!/usr/bin/env python
import gym
import numpy as np 
import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from openai_ros.task_envs.bebop2 import double_bebop2_task


if __name__ == '__main__':

    # , log_level=rospy.WARN
    rospy.init_node('double_bebop_train')

    # Create the Gym environment
    env = gym.make('DoubleBebop2Env-v0')
    rospy.logdebug("GYM ENVIRONMENT DONE")


    # Run the environment for 100 episodes
    for episode in range(100):
        # Reset the environment to its initial state
        observation = env.reset()
        total_reward = 0

        # Run the environment for 1000 steps
        for _ in range(1000):
            # Choose a random action
            action = np.random.uniform(-1,1, 4)

            # Take the action and observe the result
            observation, reward, done, info = env.step(action)
            total_reward += reward

            # If the environment is done, break the loop
            if done:
                break

        # Print the total reward for the episode
        print("Episode", episode, "Total Reward:", total_reward)

    # Close the environment
    env.close()
