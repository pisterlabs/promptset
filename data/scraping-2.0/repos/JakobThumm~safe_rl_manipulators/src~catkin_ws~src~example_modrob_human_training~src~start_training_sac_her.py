#!/usr/bin/env python
import torch
import gym
import numpy as np
import time
import rospy
import rospkg
import functools

from datetime import datetime

from spinup.algos.pytorch.sac_her import sac_her, core
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from gym import wrappers
from torch.optim import Adam
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment


if __name__ == '__main__':
    # How can this be done dynamically?
    rospy.init_node('modrob_RL_node',
                    anonymous=True, log_level=rospy.INFO)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/sac/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('example_modrob_human_training')
    now = datetime.now()
    load_epoch = -1
    if rospy.has_param("/sac/load_epoch"):
      load_epoch = rospy.get_param("/sac/load_epoch")
      outdir = pkg_path + '/training_results/' + rospy.get_param("/sac/outdir")
    else:
      outdir = pkg_path + '/training_results/' + now.strftime("%Y_%m_%d_%H_%M") 
    ## We cannot use a monitor if we want to cut off trajectories
    #env = wrappers.Monitor(env, outdir, force=True)
    #rospy.loginfo("Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    # Network size
    hid = rospy.get_param("/sac/hid")
    l = rospy.get_param("/sac/l")
    ac_kwargs=dict(hidden_sizes=hid*l)
    # Random seed
    seed = rospy.get_param("/sac/seed")
    # An epoch consists of a fixed amount of episodes
    n_episodes_per_epoch = rospy.get_param("/sac/n_episodes_per_epoch")
    # We train for a fixed amount of epochs
    n_epochs = rospy.get_param("/sac/n_epochs")
    # Size of replay buffer
    replay_size = rospy.get_param("/sac/replay_size")
    # Discount factor. (Always between 0 and 1.)
    gamma = rospy.get_param("/sac/gamma")
    # polyak (float): Interpolation factor in polyak averaging for target networks.
    polyak = rospy.get_param("/sac/polyak")
    # learning rate
    lr = rospy.get_param("/sac/lr")
    # Entropy regularization coefficient.
    alpha = rospy.get_param("/sac/alpha")
    # Batch size
    batch_size = rospy.get_param("/sac/batch_size")
    # Number of steps for uniform-random action selection,
    # before running real policy. Helps exploration.
    start_steps = rospy.get_param("/sac/start_steps")
    # Number of env interactions to collect before starting to do gradient descent updates. 
    # Ensures replay buffer is full enough for useful updates.
    update_after = rospy.get_param("/sac/update_after")
    # Number of env interactions that should elapse between gradient descent updates. Note: Regardless of how long 
    #  you wait between updates, the ratio of env steps to gradient steps is locked to 1.
    update_every = rospy.get_param("/sac/update_every")
    # Number of episodes to test the deterministic policy at the end of each epoch.
    num_test_episodes = rospy.get_param("/sac/num_test_episodes")
    # maximum length of episode
    max_ep_len = rospy.get_param("/sac/max_ep_len")
    # Number of epochs between each policy/value function save
    save_freq = rospy.get_param("/sac/save_freq")
    # Number of HER transitions per real transition
    k_her_samples = rospy.get_param("/sac/k_her_samples")
    # Number of updates steps per update
    n_updates = rospy.get_param("/sac/n_updates")

    logger_kwargs = setup_logger_kwargs(task_and_robot_environment_name,seed,outdir)

    # Set max timestep
    env.spec.timestep_limit = max_ep_len

    sac_her.sac_her(env=env, 
        test_env = env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=hid*l),
        seed=seed, 
        n_epochs=n_epochs, 
        n_episodes_per_epoch=n_episodes_per_epoch, 
        replay_size=replay_size, 
        gamma=gamma, 
        polyak=polyak, 
        lr=lr, 
        alpha=alpha, 
        batch_size=batch_size, 
        start_steps=start_steps, 
        update_after=update_after, 
        update_every=update_every, 
        num_test_episodes=num_test_episodes, 
        max_ep_len=max_ep_len, 
        n_updates = n_updates,
        k_her_samples=k_her_samples,
        logger_kwargs=logger_kwargs, 
        save_freq=save_freq,
        load_epoch=load_epoch)
    env.close()
