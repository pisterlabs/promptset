#!/usr/bin/env python3


import gym
import numpy as np
import time

from gym import wrappers
from scripts.hac.run_HAC import train_HAC, test_HAC 
from scripts.hac.agent import Agent
#ros packages
import rospy
import rospkg

#import openai_ros
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

if __name__ == '__main__':
    
    rospy.init_node('abb_world_hac', anonymous=True, log_level=rospy.WARN)

    #Init OpenAI_ROS ENV
    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/HAC_ABB/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)    
    agent = Agent(env)
    rospy.loginfo("Gym env load done")


    #Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('abb_openai_hac')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper Started")

    #load parameters from the ROS param server
    #parameters store in a yaml file inside the config directory
    Alpha = rospy.get_param("/HAC_ABB/alpha")
    Epsilon = rospy.get_param("/HAC_ABB/epsilon")
    Gamma = rospy.get_param("/HAC_ABB/gamma")
    epsilon_discount = rospy.get_param("/HAC_ABB/epsilon_discount")
    nepisodes = rospy.get_param("/HAC_ABB/nepisodes")
    nsteps = rospy.get_param("/HAC_ABB/nsteps")
    running_step = rospy.get_param("/HAC_ABB/running_step")

    # begin testing
    rospy.loginfo("Starting Testing")
    test_HAC(Alpha, Epsilon, Gamma, epsilon_discount, nepisodes, nsteps, running_step)
