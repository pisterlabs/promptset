import numpy as np
import gym
import os, sys
from mpi4py import MPI
import pdb
from HER.arguments import get_args
# from HER.rl_modules.ddpg_agent import ddpg_agent
# from HER.rl_modules.model_normed_ddpg_agent import ddpg_agent
# from HER.rl_modules.sac_agent import ddpg_agent
# from HER.rl_modules.p2p_agent import ddpg_agent
from HER.rl_modules.sac_models import StateValueEstimator

import random
import torch

from pomp.planners.plantogym import PlanningEnvGymWrapper, KinomaticGymWrapper

from pomp.example_problems.doubleintegrator import doubleIntegratorTest
from pomp.example_problems.pendulum import pendulumTest
from pomp.example_problems.gym_pendulum_baseenv import PendulumGoalEnv

from pomp.example_problems.robotics.fetch.reach import FetchReachEnv
from pomp.example_problems.robotics.fetch.push import FetchPushEnv
from pomp.example_problems.robotics.fetch.slide import FetchSlideEnv
from pomp.example_problems.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv

from pomp.example_problems.robotics.hand.reach import HandReachEnv

# from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment, EnvironmentCollection, Obstacle
from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment#, EnvironmentCollection, Obstacle

from HER_mod.rl_modules.velocity_env import MultiGoalEnvironment, CarEnvironment
from HER_mod.rl_modules.car_env import *

import pickle

from gym.wrappers.time_limit import TimeLimit
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
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    # create the ddpg_agent
    if args.env_name == "MultiGoalEnvironment":
        env = MultiGoalEnvironment("MultiGoalEnvironment", time=True, vel_goal=False)
    elif args.env_name == "MultiGoalEnvironmentVelGoal":
        env = MultiGoalEnvironment("MultiGoalEnvironment", time=True, vel_goal=True)
    elif "Car" in args.env_name:
        env = CarEnvironment("CarEnvironment", time=True, vel_goal=False)
        # env = TimeLimit(CarEnvironment("CarEnvironment", time=True, vel_goal=False), max_episode_steps=50)
    elif args.env_name == "Asteroids" :
        env = TimeLimit(RotationEnv(vel_goal=False), max_episode_steps=50)
    elif args.env_name == "AsteroidsVelGoal" :
        env = TimeLimit(RotationEnv(vel_goal=True), max_episode_steps=50)
    elif args.env_name == "PendulumGoal":
        env = TimeLimit(PendulumGoalEnv(g=9.8), max_episode_steps=200)
    elif args.env_name == "FetchReach":
        env = TimeLimit(FetchReachEnv(), max_episode_steps=50)
    elif args.env_name == "FetchPush":
        env = TimeLimit(FetchPushEnv(), max_episode_steps=50)
    elif args.env_name == "FetchSlide":
        env = TimeLimit(FetchSlideEnv(), max_episode_steps=50)
    elif args.env_name == "FetchPickAndPlace":
        env = TimeLimit(FetchPickAndPlaceEnv(), max_episode_steps=50)
    elif args.env_name == "HandReach":
        env = TimeLimit(HandReachEnv(), max_episode_steps=10)
    else:
        env = gym.make(args.env_name)

    # env = TimeLimit(FetchReachEnv(), max_episode_steps=50)
    # env = TimeLimit(FetchPushEnv(), max_episode_steps=50)
    # # problem = doubleIntegratorTest()
    # problem = pendulumTest()
    # # env = PlanningEnvGymWrapper(problem)
    # env = KinomaticGymWrapper(problem)
    # env = gym.make(args.env_name)
    # set random seeds for reproduce
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

    if args.p2p: 
        # if "Fetch" in args.env_name:
        #     from HER.rl_modules.fetch_specific_p2p import ddpg_agent
        # else: 
        #     from HER.rl_modules.p2p_agent import ddpg_agent
        from HER.rl_modules.tdm_p2p import ddpg_agent
        suffix = "_p2p"
    else: 
        # from HER.rl_modules.sac_agent import ddpg_agent
        # from HER.rl_modules.value_prior_agent import ddpg_agent
        # from HER.rl_modules.ddpg_agent import ddpg_agent
        print("Using the model from the transferred section")
        from HER_RFF_SF.rl_modules.ddpg_original import ddpg_agent
        # from HER.rl_modules.tdm_agent import ddpg_agent
        # from HER.rl_modules.tdm_ddpg_agent import ddpg_agent
        suffix = ""

    agent = launch(args)

    with open("saved_models/her_" + args.env_name + suffix + ".pkl", 'wb') as f:
        pickle.dump(agent.actor_network, f)
        print("Saved agent")

    value_estimator = StateValueEstimator(agent.actor_network, agent.critic_network, args.gamma)

    with open("saved_models/her_" + args.env_name + "_value" + suffix + ".pkl", 'wb') as f:
        pickle.dump(value_estimator, f)
        print("Saved value estimator")

