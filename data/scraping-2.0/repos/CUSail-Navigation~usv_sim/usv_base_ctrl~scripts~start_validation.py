#!/usr/bin/env python3

import gym
import matplotlib.pyplot as plt
import numpy
import time
import torch
from torch import nn, optim
from gym import wrappers
from functools import reduce
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from collections import deque
import random
from copy import deepcopy
import os

if __name__ == '__main__':

    rospy.init_node('sailboat_learn', anonymous=True, log_level=rospy.INFO)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/sailboat/training/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('usv_sim')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # path to store results
    result_dir = rospy.get_param('/sailboat/training/path_to_results')

    # Existing actor model to start from
    load_actor = rospy.get_param('/sailboat/training/load_models/load_actor')

    hidden_size1 = rospy.get_param("/sailboat/training/hidden_size1")
    hidden_size2 = rospy.get_param("/sailboat/training/hidden_size2")

    print_every = rospy.get_param("/sailboat/training/print_every")

    nepisodes = rospy.get_param("/sailboat/training/nepisodes")

    # Use GPU if possible
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        rospy.loginfo("Using device {}".format(torch.cuda.get_device_name(0)))
    else:
        rospy.logwarn("No GPU detected. Using CPU is not recommended.")
        device = torch.device("cpu")

    class NormalizedEnv(gym.ActionWrapper):

        def action(self, action):
            act_k = (self.action_space.high - self.action_space.low) / 2.
            act_b = (self.action_space.high + self.action_space.low) / 2.
            return act_k * action + act_b

        def reverse_action(self, action):
            act_k_inv = 2. / (self.action_space.high - self.action_space.low)
            act_b = (self.action_space.high + self.action_space.low) / 2.
            return act_k_inv * (action - act_b)

    def fanin_(size):
        fan_in = size[0]
        weight = 1. / numpy.sqrt(fan_in)
        return torch.Tensor(size).uniform_(-weight, weight)

    class Actor(nn.Module):

        def __init__(self,
                     state_dim,
                     action_dim,
                     h1=hidden_size1,
                     h2=hidden_size2,
                     init_w=0.003):
            super(Actor, self).__init__()

            self.linear1 = nn.Linear(state_dim, h1)
            self.linear1.weight.data = fanin_(self.linear1.weight.data.size())

            self.ln1 = nn.LayerNorm(h1)

            self.linear2 = nn.Linear(h1, h2)
            self.linear2.weight.data = fanin_(self.linear2.weight.data.size())

            self.ln2 = nn.LayerNorm(h2)

            self.linear3 = nn.Linear(h2, action_dim)
            self.linear3.weight.data.uniform_(-init_w, init_w)

            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()

        def forward(self, state):
            x = self.linear1(state)
            x = self.ln1(x)
            x = self.relu(x)

            x = self.linear2(x)
            x = self.ln2(x)
            x = self.relu(x)

            x = self.linear3(x)
            x = self.tanh(x)
            return x

        def get_action(self, state):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.forward(state)
            return action.detach().cpu().numpy()[0]

    # training
    torch.manual_seed(-1)

    env = NormalizedEnv(env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    rospy.loginfo("State dim: {}, Action dim: {}".format(
        state_dim, action_dim))

    actor = Actor(state_dim, action_dim).to(device)

    if load_actor != "":
        actor.load_state_dict(torch.load(os.path.join(result_dir, load_actor)))
    else:
        rospy.logerr("No actor provided. Using random.")

    # set eval mode
    actor.eval()

    plot_reward = []
    plot_steps = []

    average_reward = 0
    global_step = 0
    for episode in range(nepisodes):
        s = deepcopy(env.reset())

        ep_reward = 0.
        step = 0

        terminal = False
        while not terminal:
            global_step += 1
            step += 1
            a = actor.get_action(s)

            a = numpy.clip(a, -1, 1)  # normalization by normalized env wrapper
            s2, reward, terminal, info = env.step(a)

            s = deepcopy(s2)
            ep_reward += reward

        plot_reward.append(ep_reward)
        plot_steps.append(step + 1)

        average_reward += ep_reward

        if (episode % print_every) == (print_every - 1):
            # make sure path to results exists
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].set_title("Episode Rewards")
            axs[0].plot(plot_reward, 'g-')
            axs[1].set_title('Steps per Episode')
            axs[1].plot(plot_steps, 'b-')
            plt.tight_layout()

            # make sure path to results exists
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            plt.savefig(
                os.path.join(result_dir, 'validation-{}.pdf'.format(episode)))

            rospy.loginfo(
                '[%6d episode, %8d total steps] average reward for past {} iterations: %.3f'
                .format(print_every) %
                (episode + 1, global_step, average_reward / print_every))

            average_reward = 0

    env.close()