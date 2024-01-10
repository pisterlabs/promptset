import numpy as np
import gym
from datetime import datetime
import os, sys
from arguments import get_args
from mpi4py import MPI
# from tensorboardX import SummaryWriter
from rl_modules.adversarial_double_stack_sac_agent import adversarial_double_stack_sac_agent
import env_resource.envs.robotics

import random
import torch

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
"""


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              'attention_head': args.attention_head,
              'hidden_unit': args.hidden_unit,
              'grip':3 if args.only_gripper else 10,
              # 'n_object':env.n_object,
              'device': args.device,
              }
    # params['max_timesteps'] = env._max_episode_steps
    params['max_timesteps'] = args.max_episode_steps
    return params


def launch(args):
    # create the ddpg_agent
    env = gym.make(args.env_name)
    # env = gym.make('FetchPickAndPlace-v1')
    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment
    writer = None
    # if args.tensorboard is True and MPI.COMM_WORLD.Get_rank() == 0:
    #     current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    #     log_path = args.logdir + '_' + current_time
    #     if not os.path.exists(log_path):
    #         writer = SummaryWriter(log_path)

    if args.algo == 'adversarial_double_pick_sac':
        trainer = adversarial_double_stack_sac_agent(args, env, env_params, writer)
    else:
        raise NotImplementedError

    trainer.visualize_random_noise_test(
        'saved_models/MultiFetchStackEasyEnv-v1/sac/adv_sac_double_pick.pt')
    # upper path for adversary, lower path for protagonist

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
