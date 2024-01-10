#!/usr/bin/env python
import gym
# import numpy as np
import rospy
import sac

from openai_ros.task_envs.bebop2 import double_bebop2_task

def make_env():
    return gym.make('DoubleBebop2Env-v0')


rospy.init_node("train_double_bebop_sac")

sac.sac(make_env, load_path='/home/huss/.ros/Models_sac/51000', episode= 51000, start_steps=0)










