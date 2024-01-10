import numpy as np
import gym 
import os, sys 
from mpi4py import MPI 
import pdb 
from HER.arguments import get_args 

import random
import torch

sys.path.append('./pybullet_env')
# from pybullet_env import environment, utilities, robot
# from environment import make_throwing_env
from save_state_env import make_throwing_env
# from pybullet_env.utilities import *


# from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment, EnvironmentCollection, Obstacle
# from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment#, EnvironmentCollection, Obstacle

# from environments.velocity_env import MultiGoalEnvironment, CarEnvironment
# from environments.car_env import RotationEnv, NewCarEnv, SimpleMovementEnvironment
# from environments.torus_env import Torus
# from environments.continuous_acrobot import ContinuousAcrobotEnv

import pickle

from gym.wrappers.time_limit import TimeLimit
from action_randomness_wrapper import ActionRandomnessWrapper, RepeatedActionWrapper
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
    # create the ddpg_agent
    if args.env_name == "MultiGoalEnvironment":
        env = MultiGoalEnvironment("MultiGoalEnvironment", time=True, vel_goal=False)
    elif args.env_name == "MultiGoalEnvironmentVelGoal":
        env = MultiGoalEnvironment("MultiGoalEnvironment", time=True, vel_goal=True)
    elif "Car" in args.env_name:
        # env = CarEnvironment("CarEnvironment", time=True, vel_goal=False)
        env = TimeLimit(NewCarEnv(vel_goal=False), max_episode_steps=50)
        # env = TimeLimit(CarEnvironment("CarEnvironment", time=True, vel_goal=False), max_episode_steps=50)
    elif args.env_name == "Asteroids" :
        env = TimeLimit(RotationEnv(vel_goal=False), max_episode_steps=50)
    elif args.env_name == "AsteroidsVelGoal" :
        env = TimeLimit(RotationEnv(vel_goal=True), max_episode_steps=50)
    elif "SimpleMovement" in args.env_name:
        env = TimeLimit(SimpleMovementEnvironment(vel_goal=False), max_episode_steps=50)
    elif args.env_name == "PendulumGoal":
        env = TimeLimit(PendulumGoalEnv(g=9.8), max_episode_steps=200)
    elif args.env_name == "Throwing":
        env = TimeLimit(make_throwing_env(), max_episode_steps=20)
    elif "Gridworld" in args.env_name: 
        # from continuous_gridworld import create_map_1#, random_blocky_map, two_door_environment, random_map
        # from alt_gridworld_implementation import create_test_map, random_blocky_map, two_door_environment, random_map #create_map_1,
        from environments.solvable_gridworld_implementation import create_test_map, random_blocky_map, two_door_environment, random_map #create_map_1,
        # from gridworld_reimplementation import random_map

        max_steps = 50 if "Alt" in args.env_name else 20
        if args.env_name == "TwoDoorGridworld":
            env=TimeLimit(two_door_environment(), max_episode_steps=50)
        else:
            if "RandomBlocky" in args.env_name:
                mapmaker = random_blocky_map
            elif "Random" in args.env_name:
                mapmaker = random_map
            elif "Test" in args.env_name: 
                mapmaker = create_test_map
            else: 
                mapmaker = create_map_1

            if "Asteroids" in args.env_name: 
                env_type="asteroids"
            elif "StandardCar" in args.env_name:
                env_type = "standard_car"
            elif "Car" in args.env_name:
                env_type = "car"
            else: 
                env_type = "linear"
            print(f"env type: {env_type}")
            env = TimeLimit(mapmaker(env_type=env_type), max_episode_steps=max_steps)
    elif "ContinuousAcrobot" in args.env_name:
        env = TimeLimit(ContinuousAcrobotEnv(), max_episode_steps=50)
    elif "2DNav" in args.env_name or "2Dnav" in args.env_name: 
        env = gym.make("Limited-Range-Based-Navigation-2d-Map8-Goal0-v0")
    else:
        env = gym.make(args.env_name)

    env = ActionRandomnessWrapper(env, args.action_noise)

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

    if args.resume == True:
        ddpg_trainer.resume_learning()
    else:
        ddpg_trainer.learn()

    # ddpg_trainer.learn()
    return ddpg_trainer

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    # print(args)
    # exit()

    # from HER.rl_modules.generalized_usher_with_ratio_2 import ddpg_agent
    from HER.rl_modules.generalized_usher_resume_test import ddpg_agent
    suffix = ""

    agent = launch(args)

    
    n = 10
    # success_rate, reward, value = ev['success_rate'], ev['reward_rate'], ev['value_rate']
    success_rate = sum([agent._eval_agent(final=True)['success_rate'] for _ in range(n)])/n
    if LOGGING and MPI.COMM_WORLD.Get_rank() == 0:
        # pdb.set_trace()
        log_file_name = f"logging/{args.env_name}.txt"
        # success_rate = sum([agent._eval_agent()[0] for _ in range(n)])/n
        text = f"action_noise: {args.action_noise}, \ttwo_goal: {args.two_goal}, \tsuccess_rate: {success_rate}\n"
        with open(log_file_name, "a") as f:
            f.write(text)

        print("Log written")
