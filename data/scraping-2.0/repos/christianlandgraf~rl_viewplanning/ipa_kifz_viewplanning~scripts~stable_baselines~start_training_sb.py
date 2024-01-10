#!/usr/bin/env python
import gym
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from geometry_msgs.msg import Pose

from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment_Parallel
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from openai_ros.msg import RLExperimentInfo
from stable_baselines3 import PPO, DQN
from stable_baselines3.dqn import MlpPolicy as DQNMlpPolicy
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, register_policy
from stable_baselines3.common.utils import get_schedule_fn
# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import time
import datetime
import os
import sys
import numpy as np
from typing import Callable

import torch
from tensorboardX import SummaryWriter

if __name__ == '__main__':

    #############################
    # Initialization
    #############################
    rospy.init_node('ipa_kifz_viewplanning_sb',
                    anonymous=True,
                    log_level=rospy.WARN)

    # Init OpenAI_ROS Environment
    rospy.loginfo("Init Task Environment")
    task_and_robot_environment_name = rospy.get_param(
        '/ipa_kifz/task_and_robot_environment_name')
    rospy.loginfo("Init ROS Environment")

    algorithm = rospy.get_param("/algorithm")

    #############################
    # Parameter
    #############################

    # Common parameters
    policy = rospy.get_param('/ipa_kifz/policy')
    learning_rate = rospy.get_param('/ipa_kifz/learning_rate')
    batch_size = rospy.get_param('/ipa_kifz/batch_size')

    total_timesteps = rospy.get_param('/ipa_kifz/total_timesteps')
    num_envs = rospy.get_param('ipa_kifz/num_envs', 1)

    # Algorithm-specific parameters
    if algorithm == "dqn":
        buffer_size = rospy.get_param('/ipa_kifz/buffer_size')
        learning_starts = rospy.get_param('/ipa_kifz/learning_starts')
        tau = rospy.get_param('/ipa_kifz/tau')
        gamma = rospy.get_param('/ipa_kifz/gamma')
        train_freq = rospy.get_param('/ipa_kifz/train_freq')
        gradient_steps = rospy.get_param('/ipa_kifz/gradient_steps')
        n_episodes_rollout = rospy.get_param('/ipa_kifz/n_episodes_rollout')
        target_update_interval = rospy.get_param(
            '/ipa_kifz/target_update_interval')
        exploration_fraction = rospy.get_param(
            '/ipa_kifz/exploration_fraction')
        exploration_initial_eps = rospy.get_param(
            '/ipa_kifz/exploration_initial_eps')
        exploration_final_eps = rospy.get_param(
            '/ipa_kifz/exploration_final_eps')
        max_grad_norm = rospy.get_param('/ipa_kifz/max_grad_norm')
    elif algorithm == "ppo":
        n_steps = rospy.get_param("/ipa_kifz/n_steps")
        batch_size = rospy.get_param("/ipa_kifz/batch_size")
        n_epochs = rospy.get_param("/ipa_kifz/n_epochs")
        gamma = rospy.get_param("/ipa_kifz/gamma")
        gae_lambda = rospy.get_param("/ipa_kifz/gae_lambda")
        clip_range = rospy.get_param("/ipa_kifz/clip_range")
        ent_coef = rospy.get_param("/ipa_kifz/ent_coef")
        vf_coef = rospy.get_param("/ipa_kifz/vf_coef")
        max_grad_norm = rospy.get_param("/ipa_kifz/max_grad_norm")

    #############################
    # Logging
    #############################

    logfile_time = datetime.datetime.now()
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ipa_kifz_viewplanning')

    sys.path.append(os.path.join(pkg_path, "scripts"))
    from helper import ipa_kifz_logging
    sbLogger = ipa_kifz_logging.Logger(logfile_time, log_freq=1)
    sbLogger.save_params()

    writer = SummaryWriter(sbLogger.log_directory)

    def pose_callback(data):
        sbLogger.log_pose([
            data.position.x, data.position.y, data.position.z,
            data.orientation.x, data.orientation.y, data.orientation.z,
            data.orientation.w
        ])

    def reward_callback(data):
        sbLogger.log_episode(data.episode_reward)
        writer.add_scalar('episode_reward', data.episode_reward,
                          data.episode_number)

    pose_sub = rospy.Subscriber("openai/episode_poses", Pose, pose_callback)
    reward_sub = rospy.Subscriber("openai/reward", RLExperimentInfo,
                                  reward_callback)

    #############################
    # Gym Environment Stuff
    #############################


    def create_parallel_envs(env_id, env_name):
        """
        Helper function for creating parallel environments for training
        """
        eid = env_id + 1
        print("eid: " + str(eid))
        env = StartOpenAI_ROS_Environment_Parallel(env_name, eid)
        return env

    envs = []

    def ret_lambda_func(k, name):
        return lambda: create_parallel_envs(k, name)

    for k in range(num_envs):
        envs.append(ret_lambda_func(k, task_and_robot_environment_name))

    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

    # env = make_vec_env('IPA_KIFZ_Viewplanning-v0', n_envs=1)

    print("Gym environment done")

    #############################
    # Learning Rate Schedule
    #############################


    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:

            return progress_remaining * initial_value

        return func

    #############################
    # Setup Model
    #############################

    if algorithm == "dqn":
        # policy = DQNMlpPolicy(
        #     env.observation_space,
        #     env.action_space,
        #     lr_scheduler,
        #     net_arch=[64, 64],
        #     optimizer_class = optimizer
        # )

        model = DQN(
            DQNMlpPolicy,  #policy #MlpPolicy
            env,
            learning_rate=linear_schedule(learning_rate),
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            n_episodes_rollout=n_episodes_rollout,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=sbLogger.log_directory,
            verbose=2)

    elif algorithm == "ppo":
        # custom_ppo_mlppolicy = PPOMlpPolicy(
        #     env.observation_space,
        #     env.action_space,
        #     linear_schedule(learning_rate),
        # )

        # Define the RL algorithm and start learning
        model = PPO(
            PPOMlpPolicy,
            env,
            learning_rate=learning_rate,  #linear_schedule(0.1),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=None,  # - depends on reward scaling!
            ent_coef=ent_coef,
            tensorboard_log=sbLogger.log_directory,
            verbose=2,
        )

    sbLogger.model = model
    sbLogger.training_env = model.get_env()

    #############################
    # Learning
    #############################
    '''Training the model'''
    rospy.logwarn("Start training")
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=sbLogger.log_freq * 4,
        callback=[CheckpointCallback(1000, sbLogger.log_directory)],
        # tb_log_name='DQN',
    )

    sbLogger.save_model(model)
    rospy.logwarn("DONE TRAINING")

    # Load saved model and make predictions
    # del model
    # model = DQN.load("ipa_kifz_viewplanning-v0")
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

    writer.close()
    env.close()