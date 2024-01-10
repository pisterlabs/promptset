import numpy as np
import gym 
import os, sys 
from mpi4py import MPI 
import pdb 
from HER_mod.arguments import get_args 

import random
import torch

sys.path.append('./pybullet_env')
# from pybullet_env import environment, utilities, robot
# from environment import make_throwing_env
# from pybullet_env.utilities import *


# from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment, EnvironmentCollection, Obstacle
# from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment#, EnvironmentCollection, Obstacle

# from environments.velocity_env import MultiGoalEnvironment, CarEnvironment
# from environments.car_env import RotationEnv, NewCarEnv, SimpleMovementEnvironment
# from environments.torus_env import Torus
# from environments.continuous_acrobot import ContinuousAcrobotEnv

import pickle

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""

LOGGING = True


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    from env_maker import make_env
    env = make_env(args)

    try: 
        env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    except: 
        pass
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment 


    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()
    return ddpg_trainer

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
 
    from HER_mod.rl_modules.usher_agent_2d import ddpg_agent   
    suffix = ""

    agent = launch(args)

    
    n = 10
    evs = [agent._eval_agent() for _ in range(n)]
    success_rate = sum([evs[i]['success_rate'] for i in range(n)])/n
    reward_rate = sum([evs[i]['reward_rate'] for i in range(n)])/n
    value_rate = sum([evs[i]['value_rate'] for i in range(n)])/n
    if LOGGING and MPI.COMM_WORLD.Get_rank() == 0:
        log_file_name = f"logging/{args.env_name}.txt"
        text = f"action_noise: {args.action_noise}, "   
        text +=f"\ttwo_goal: {args.two_goal}, \n"            
        text +=f"\tsuccess_rate: {success_rate}\n"         
        text +=f"\taverage_reward: {reward_rate}\n"        
        text +=f"\taverage_initial_value: {value_rate}\n"  
        text +="\n"

        with open(log_file_name, "a") as f:
            f.write(text)

        print("Log written")
