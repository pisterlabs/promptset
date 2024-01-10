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
#import arguments
from scripts.args import Args

def main():

    rospy.init_node('abb_world_hac', anonymous=True, log_level=rospy.WARN)
    


    #load parameters from the ROS param server
    #parameters store in a yaml file inside the config directory
    task_and_robot_environment_name = rospy.get_param(
        '/HAC_ABB/task_and_robot_environment_name')

    lr = rospy.get_param("/HAC_ABB/lr")
    Epsilon = rospy.get_param("/HAC_ABB/epsilon")
    Gamma = rospy.get_param("/HAC_ABB/gamma")
    epsilon_discount = rospy.get_param("/HAC_ABB/epsilon_discount")
    nepisodes = rospy.get_param("/HAC_ABB/nepisodes")
    running_step = rospy.get_param("/HAC_ABB/running_step")
    retrain = rospy.get_param("/HAC_ABB/retrain")
    test = rospy.get_param("/HAC_ABB/test")
    show = rospy.get_param("/HAC_ABB/show")
    train_only = rospy.get_param("/HAC_ABB/train_only")
    verbose = rospy.get_param("/HAC_ABB/verbose1")
    log_level = rospy.get_param("/HAC_ABB/log_level")
    n_layers = rospy.get_param("/HAC_ABB/n_layer")
    num_batch_train = rospy.get_param("/HAC_ABB/num_batch_train")
    timesteps = rospy.get_param("/HAC_ABB/timesteps")
    test_freq = rospy.get_param("/HAC_ABB/test_freq")
    seed = rospy.get_param("/HAC_ABB/seed")



    #reconstuct param
    args = Args()
    """lr,
    Epsilon, 
    Gamma, 
    epsilon_discount, 
    nepisodes,
    running_step,
    retrain,
    test,
    show,
    train_only,
    verbose,
    log_level,
    n_layers,
    num_batch_train,
    timesteps,
    test_freq,
    seed"""


    # Init OpenAI_ROS ENV
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name, args)

    
    rospy.loginfo("Gym env load done")

    #Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('abb_openai_hac')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper Started")

  


    agent = Agent(env, args)
    # begin training
    train_HAC(args, env, agent)

if __name__ == '__main__':
    
    main()
