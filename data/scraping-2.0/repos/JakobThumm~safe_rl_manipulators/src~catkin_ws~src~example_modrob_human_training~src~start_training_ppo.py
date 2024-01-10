#!/usr/bin/env python
import torch
import gym
import numpy as np
import time
import qlearn
import rospy
import rospkg
import functools

from spinup.algos.pytorch.ppo import ppo, core
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
        '/ppo/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('example_modrob_human_training')
    outdir = pkg_path + '/training_results'
    ## We cannot use a monitor if we want to cut off trajectories
    #env = wrappers.Monitor(env, outdir, force=True)
    #rospy.loginfo("Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    alpha = rospy.get_param("/ppo/alpha")
    gamma = rospy.get_param("/ppo/gamma")
    # An episode is defined by the environment. The robot starts in initial state and runs until done.
    # Each episode consists of max max_steps_per_episode steps.
    max_steps_per_episode = rospy.get_param("/ppo/max_steps_per_episode")
    # We train for a fixed amount of epochs
    n_epochs = rospy.get_param("/ppo/n_epochs")
    # An epoch consists of a fixed amount of steps.
    steps_per_epoch = rospy.get_param("/ppo/steps_per_epoch")
    
    running_step = rospy.get_param("/ppo/running_step")
    seed = rospy.get_param("/ppo/seed")
    hid = rospy.get_param("/ppo/hid")
    l = rospy.get_param("/ppo/l")
    clip_ratio = rospy.get_param("/ppo/clip_ratio")
    pi_lr = rospy.get_param("/ppo/pi_lr")
    vf_lr = rospy.get_param("/ppo/vf_lr")
    train_pi_iters = rospy.get_param("/ppo/train_pi_iters")
    train_v_iters = rospy.get_param("/ppo/train_v_iters")
    lam = rospy.get_param("/ppo/lam")
    target_kl = rospy.get_param("/ppo/target_kl")
    logger_kwargs=dict()
    save_freq = rospy.get_param("/ppo/save_freq")

    # Set max timestep
    env.spec.timestep_limit = max_steps_per_episode

    ppo.ppo(env=env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=hid*l),
        seed=seed, 
        steps_per_epoch=steps_per_epoch, 
        epochs=n_epochs, 
        gamma=gamma, 
        clip_ratio=clip_ratio, 
        pi_lr=pi_lr,
        vf_lr=vf_lr, 
        train_pi_iters=train_pi_iters, 
        train_v_iters=train_v_iters, 
        lam=lam, 
        max_ep_len=max_steps_per_episode,
        target_kl=target_kl, 
        logger_kwargs=logger_kwargs, 
        save_freq=save_freq)
    env.close()
