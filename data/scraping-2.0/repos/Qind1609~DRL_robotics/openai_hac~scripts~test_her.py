#! /usr/bin/env python3

import random
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import rospy
import torch
import os
from her.ddpg_her_normalization import *
import numpy as np
from mpi4py import MPI


class Hyper_Params:
    def __init__(self):
        self.env_name = "NachiReach_v0"
        self.seed = 362514 

        # number of epochs for training
        self.num_epochs = 100

        # number of episodes - the times to collect samplers per epoch (reset -> new goal ->action)
        self.num_episodes = 50

        # maximum step for 1 episode
        self.max_ep_step = 300  # steps

        # the times to update networks
        self.num_batches = 50  # (divide the dataset into 50 batch, shuffle and random sample for each batch)

        # batch size
        self.batch_size = 300

        # initial number of step for random exploration
        # self.start_steps = 10000

        # size of replay buffer
        self.buff_size = 1000000  #  buffer size => 1000000 transitions

        # test phase
        self.phase = "test"

        # path to save model
        self.save_dir = (
            "/home/qind/Desktop/catkin_ws/src/openai_hac/scripts/her/saved_models"
        )

        # number of episodes testing should run
        self.test_episodes = 100

        # the clip ratio
        self.clip_obs = np.inf

        # the clip range
        self.clip_range = np.inf

        # learning rate actor
        self.lr_actor = 0.001

        # learning rate critic
        self.lr_critic = 0.001

        # scaling factor for gausian noise on action
        self.noise_eps = 0.1

        # random epsilon
        self.random_eps = 0.3

        # discount factor in bellman equation
        self.gamma = 0.98

        # polyak value for averaging
        self.polyak = 0.95

        # cuda - using GPU?
        self.cuda = True

        # number of worker (load data from cpu to gpu)
        # self.num_workers = 1

        # the rollout per MPI
        self.num_rollouts_per_mpi = 2

        # l2 regularization
        self.action_l2 = 1

        # replay_k
        self.replay_k = 4

        # threshold success
        self.threshold = 0.005

        # training space
        self.position_x_max = 0.63
        self.position_x_min = 0.3
        self.position_y_max = 0.145
        self.position_y_min = -0.145
        self.position_z_max = 0.31
        self.position_z_min = 0.15


def test_HER():

    params = Hyper_Params()
    rospy.init_node("HER_reach")
    task_and_robot_environment_name = rospy.get_param(
        "/Nachi/task_and_robot_environment_name"
    )

    # start env
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name, params.max_ep_step
    )

    # set seed - make sure model give same result every run
    env.seed(params.seed + MPI.COMM_WORLD.Get_rank())
    env.action_space.seed(params.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(params.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(params.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(params.seed + MPI.COMM_WORLD.Get_rank())
    if params.cuda:
        torch.cuda.manual_seed(params.seed + MPI.COMM_WORLD.Get_rank())
    
    # reset and get observation
    obs = env.reset()

    env_params = {
        "obs_dim": obs["observation"].shape[0],
        "goal_dim": obs["desired_goal"].shape[0],
        "action_dim": env.action_space.shape[0],
        "action_max": env.action_space.high[0],
        "max_timesteps": env._max_episode_steps,  # max_step for each ep
    }

    ddpg_agent = Test_DDPG_HER(params, env, env_params)
    ddpg_agent.test()
    
    rospy.logwarn(
        "####################### Testing Complete ##########################"
    )


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["IN_MPI"] = "1"

    test_HER()
