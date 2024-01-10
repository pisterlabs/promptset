#!/usr/bin/env python

import gym
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from functools import reduce
import pickle



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import math
import glob
import io
import base64
import datetime
import json




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



def test(env, policy_net, device, test_global_step, render=False):
    state, ep_reward, done = env.reset(), 0, False
    state = [round(num, 1) for num in state]
    rospy.logwarn("Entering test method...")
    test_local_step = 0
    while not done:
        if render:
            env.render()
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the largest expected reward.
        action = policy_net(state).max(dim=1)[1].view(1, 1)
        state, reward, done, _ = env.step(action.item())
        state = [round(num, 1) for num in state]
        
        test_local_step += 1
        test_global_step += 1
        rospy.logwarn("Testing: Reward of this step: ")
        rospy.logwarn(reward)
        ep_reward += reward
        rospy.logwarn("Testing: Cumulative Reward of this episode: ")
        rospy.logwarn(ep_reward)
        writer.add_scalar("Test_Cumulative_Rewards", ep_reward, global_step=test_global_step)
    return ep_reward, test_global_step



if __name__ == '__main__':

    rospy.init_node('test_turtlebot2_maze_dqn', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    #task_and_robot_environment_name = rospy.get_param('task_and_robot_environment_name')
    task_and_robot_environment_name = rospy.get_param('/turtlebot2/task_and_robot_environment_name')
    # Create the Gym environment
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Test")


    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('turtle2_openai_ros_example')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    
    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    #MODEL_PATH = rospy.get_param("model_ckpt")
    #########################################################################################
    #MODEL_PATH = '$HOME/python3_ws/src/turtle2_openai_ros_example/src/checkpoints/dqn-final-episode-2671-step-110007.pt'
    model_dir = os.path.dirname(__file__)
    #MODEL_PATH = os.path.join(model_dir, 'checkpoints/dqn-final-episode-2671-step-110007.pt')
    MODEL_PATH = os.path.join(model_dir, 'checkpoints/dqn-sparse_reward-episode-1042-step-122000.pt')
    


    """
    Alpha = rospy.get_param("/turtlebot2/alpha")
    Epsilon = rospy.get_param("/turtlebot2/epsilon")
    Gamma = rospy.get_param("/turtlebot2/gamma")
    epsilon_discount = rospy.get_param("/turtlebot2/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot2/nepisodes")
    nsteps = rospy.get_param("/turtlebot2/nsteps")
    running_step = rospy.get_param("/turtlebot2/running_step")
    """


    # Hyperparameters
    gamma = 0.79  # initially 0.99 discount factor
    seed = 543  # random seed
    n_epochs = 20 # number of epochs to test the trained model

    test_global_step = 0 # Global number of testing steps for tracking cummulative rewards in Tensorboard

    # If gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Fix random seed (for reproducibility)
    env.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Get number of actions from gym action space
    #n_inputs = env.observation_space.shape[0] 
    n_inputs = 5 
    n_actions = env.action_space.n

    policy_net = DQN(n_inputs, n_actions).to(device)
    policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy_net.eval()

    ####################################################################################################################
    #logdir = os.path.join("$HOME/python3_ws/src/turtle2_openai_ros_example/src/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    basedir = os.path.dirname(__file__)
    basedirpath = os.path.join(basedir, 'logs')
    logdir = os.path.join(basedirpath, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=logdir)

    for i in range(n_epochs):
        ep_reward, test_global_step = test(env, policy_net, device, test_global_step)
    print('Steps: {}'
          '\tTest reward: {:.2f}'.format(test_global_step, ep_reward))



    

    
