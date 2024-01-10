#!/usr/bin/env python3.6

import sys
print("in run_algo: Python version: ", sys.version)
# if sys.version_info[0] < 3:
#         raise Exception("Must be using Python 3 on ROS")
import rospy
import rospkg
import random
# from home.roboticlab14.Documents.Git.baselines.baselines import deepq
# /home/roboticlab14/Documents/Git/baselines
# import sys
# sys.path.append('/home/roboticlab14/Documents/Git/baselines/baselines')

# from openai_ros.task_envs.iiwa_tasks import iiwa_move
import gym
from gym import wrappers
from gym.envs.registration import register

import roslaunch
import os
import git
import numpy
# import sys
# import baselines #import PPO2
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines import PPO2

# from baselines import ppo2
# import importlib.util
# spec = importlib.util.spec_from_file_location("baselines", "home/roboticlab14/Documents/Git/baselines")
# foo = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(foo)
# foo.MyClass()

def start_learning():
    print("I am in script run algo")

def random_action():
    a = []
    for i in range(0, 6):
        a.append(random.uniform(-0.1, 0.1))
        # i = i+1
    a[2]=-0.05
    return a

if __name__ == '__main__':

    rospy.init_node('start_python3', anonymous=True, log_level=rospy.WARN)
    # rospy.sleep(20.0)

    # start_learning()
    # start_learning()
    start_learning()
    # start_learning()
    # start_learning()
    timestep_limit_per_episode = 10000
    max_episode = 10
    # # Train a model
    # # trained_model = ppo2('MlpPolicy', 'iiwaMoveEnv-v0').learn(total_timesteps=10000)
    # register(
    #         id="iiwaMoveEnv-v0",
    #         entry_point='openai_ros.task_envs.iiwa_tasks.iiwa_move:iiwaMoveEnv',
    #     #     timestep_limit=timestep_limit_per_episode, #old one...
    #         max_episode_steps=timestep_limit_per_episode,
    #     )

    #     # Create the Gym environment
    # env = gym.make('iiwaMoveEnv-v0')

    # # multiprocess environment
    # # n_cpu = 4
    # # env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])
    # # register(
    # #         id="iiwaMoveEnv-v0",
    # #         entry_point='openai_ros.task_envs.iiwa_tasks.iiwa_move:iiwaMoveEnv',
    # #     #     timestep_limit=timestep_limit_per_episode,
    # #         max_episode_steps=timestep_limit_per_episode,
    # #     )
    # # env = gym.make('iiwaMoveEnv-v0')
    # model = PPO2(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=25000)
    # model.save("ppo2_cartpole_bim")
    # start_learning()
    # start_learning()
    # start_learning()


    # Where I thest EVERYTHING
        # observation, reward, done, info
    # for i in range(0, 9):
    #     # raw_input("Press Enter to continue...")
    #     a=random_action()
    #     # env.step(a)
    #     observation, reward, done, info = env.step(a)
    #     print("*********************************************")
    #     print("Observation: ", observation)
    #     print("Reward: ", reward)
    #     print("Done: ", done)
    #     print("Info: ", info)
    #     print("Action: ",  a)
    #     print("*****")

    # start_learning()