import numpy as np
import gym
import os, sys
from mpi4py import MPI
import random
import torch
import itertools
# from rl_modules.multi_goal_env2 import *
from HER_mod.arguments import get_args
from HER_mod.rl_modules.ddpg_agent import ddpg_agent
# from HER_mod.rl_modules.value_prior_agent import ddpg_agent
# from HER.rl_modules.her_ddpg_agent import her_ddpg_agent
from HER_mod.rl_modules.velocity_env import *
from HER_mod.rl_modules.car_env import *
# from pomp.planners.plantogym import *
from HER_mod.rl_modules.value_map import *
from HER_mod.rl_modules.hooks import *
from HER_mod.rl_modules.models import StateValueEstimator
from HER_mod.rl_modules.tsp import *
from HER_mod.rl_modules.get_path_costs import get_path_costs, get_random_search_costs, method_comparison

from pomp.planners.plantogym import PlanningEnvGymWrapper, KinomaticGymWrapper
from pomp.example_problems.doubleintegrator import doubleIntegratorTest
from pomp.example_problems.dubins import dubinsCarTest
from pomp.example_problems.pendulum import pendulumTest
from pomp.example_problems.robotics.fetch.reach import FetchReachEnv
from pomp.example_problems.robotics.fetch.push import FetchPushEnv
from pomp.example_problems.robotics.fetch.slide import FetchSlideEnv
from pomp.example_problems.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv

from gym_extensions.continuous.gym_navigation_2d.env_generator import Environment#, EnvironmentCollection, Obstacle

from pomp.example_problems.gym_pendulum_baseenv import PendulumGoalEnv
from gym.wrappers.time_limit import TimeLimit

import pickle
from action_randomness_wrapper import ActionRandomnessWrapper


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
    # print(params)
    return params

def launch(args, time=True, hooks=[], vel_goal=False, seed=True):
    # create the ddpg_agent
    # if args.env_name == "MultiGoalEnvironment":
    #     env = MultiGoalEnvironment("MultiGoalEnvironment", time=time, vel_goal=vel_goal)
    # elif args.env_name == "PendulumGoal":
    #     env = TimeLimit(PendulumGoalEnv(g=9.8), max_episode_steps=200)
    # else:
    #     env = gym.make(args.env_name)

    if args.env_name == "MultiGoalEnvironment":
        env = MultiGoalEnvironment("MultiGoalEnvironment", time=True, vel_goal=False)
    elif args.env_name == "MultiGoalEnvironmentVelGoal":
        env = MultiGoalEnvironment("MultiGoalEnvironment", time=True, vel_goal=True)
    elif args.env_name == "Car":
        env = CarEnvironment("CarEnvironment", time=True, vel_goal=False)
        # env = TimeLimit(CarEnvironment("CarEnvironment", time=True, vel_goal=False), max_episode_steps=50)
    elif "Asteroids" in args.env_name:
        env = TimeLimit(RotationEnv(vel_goal=False), max_episode_steps=50)
    elif "AsteroidsVelGoal" in args.env_name:
        env = TimeLimit(RotationEnv(vel_goal=True), max_episode_steps=50)
    elif args.env_name == "PendulumGoal":
        env = TimeLimit(PendulumGoalEnv(g=9.8), max_episode_steps=200)
    elif "FetchReach" in args.env_name:
        env = TimeLimit(FetchReachEnv(), max_episode_steps=50)
    elif "FetchPush" in args.env_name:
        env = TimeLimit(FetchPushEnv(), max_episode_steps=50)
    elif "FetchSlide" in args.env_name:
        env = TimeLimit(FetchSlideEnv(), max_episode_steps=50)
    elif "FetchPickAndPlace" in args.env_name:
        env = TimeLimit(FetchPickAndPlaceEnv(), max_episode_steps=50)
    else:
        env = gym.make(args.env_name)


    env = ActionRandomnessWrapper(env, args.action_noise)
    # env = TimeLimit(FetchReachEnv(), max_episode_steps=50)
    # env = TimeLimit(FetchPushEnv(), max_episode_steps=50)
    # env = MultiGoalEnvironment("MultiGoalEnvironment", time=time, vel_goal=vel_goal)#, epsilon=.1/4) 
    # problem = doubleIntegratorTest()
    # problem = pendulumTest()
    # env = PlanningEnvGymWrapper(problem)
    # env = KinomaticGymWrapper(problem)
    # set random seeds for reproduce
    # if seed: 
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
    # return
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params, vel_goal=vel_goal)
    # if vel_goal: 
    #     ddpg_trainer = ddpg_agent(args, env, env_params, vel_goal=vel_goal)
    # else: 
    #     ddpg_trainer = her_ddpg_agent(args, env, env_params)
    # pdb.set_trace()
    ddpg_trainer.learn(hooks)
    # [hook.finish() for hook in hooks]
    return ddpg_trainer, [hook.finish() for hook in hooks]




if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()

    # agent = launch(args, time=False, hooks=[])#hooks=[DistancePlottingHook()])
    # agent = launch(args, time=True, hooks=[DistancePlottingHook(), PlotPathCostsHook(args)], vel_goal=True)
    # agent = launch(args, time=True, hooks=[DistancePlottingHook(), PlotPathCostsHook(args)], vel_goal=False)
    # try:
    hook_list = [
                # ValueMapHook(target_vels=[[0,0], [.5/2**.5,.5/2**.5]]),
                # DiffMapHook(), 
                # # EmpiricalVelocityValueMapHook(),
                # VelocityValueMapHook(), 
                # GradientDescentShortestPathHook(),
                # GradientDescentShortestPathHook(gd_steps=5),
                # GradientDescentShortestPathHook(gd_steps=10),
                # GradientDescentShortestPathHook(gd_steps=15),
                # GradientDescentShortestPathHook(args=([0, 5,10,20,40], False)),
                # # GradientDescentShortestPathHook(args=([0,5,10,20,40], True)),
                # PlotPathCostsHook()
                ]
    # hook_list = []
    pos_hook_list = [#DiffMapHook(), 
                ValueMapHook(target_vels=[[0,0]]),#target_vels=[[0,0], [.5/2**.5,.5/2**.5]]),
                # GradientDescentShortestPathHook(),
                # GradientDescentShortestPathHook(gd_steps=5),
                # GradientDescentShortestPathHook(gd_steps=10),
                # GradientDescentShortestPathHook(gd_steps=15),
                GradientDescentShortestPathHook(args=([-1], False)),
                # GradientDescentShortestPathHook(args=([-1], True)),
                # PlotPathCostsHook()
                ]
    vel_hook_list = [
                GradientDescentShortestPathHook(args=([0,5,10,20], False)),
                GradientDescentShortestPathHook(args=([0,5,10,20], True)),
                PlotPathCostsHook()
                ]

    # hook_list = []
    # pos_hook_list = []
    # vel_hook_list = []

    train_pos_agent = lambda : launch(args, time=True, hooks=[], vel_goal=False, seed=False)[0]
    train_vel_agent = lambda : launch(args, time=True, hooks=[], vel_goal=True, seed=False)[0]
    # get_path_costs(train_pos_agent, train_vel_agent)
    # train_pos_agent()
    # train_vel_agent()
    # for i in range(10):
    #     args.seed += 1
        # agent, run_times = launch(args, time=True, hooks=hook_list, vel_goal=True, seed=False)
    # agent, run_times = launch(args, time=True, hooks=hook_list, vel_goal=True, seed=False)

    if args.p2p: 
        # if "Fetch" in args.env_name:
        #     from HER.rl_modules.fetch_specific_p2p import ddpg_agent
        # else: 
        #     from HER.rl_modules.p2p_agent import ddpg_agent
        agent, run_times = launch(args, time=True, hooks=[], vel_goal=True, seed=False)
        suffix = "_p2p"
    else: 
        if args.two_goal:
            from HER_mod.rl_modules.usher_agent import ddpg_agent
        else:
            from HER_mod.rl_modules.ddpg_agent import ddpg_agent
        # from HER.rl_modules.sac_agent import ddpg_agent
        agent, run_times = launch(args, time=True, hooks=[], vel_goal=False, seed=False)
        suffix = ""


    # with open("saved_models/her_mod_" + args.env_name + suffix + ".pkl", 'wb') as f:
    #     pickle.dump(agent.actor_network, f)
    #     print("Saved agent")

    # value_estimator = StateValueEstimator(agent.actor_network, agent.critic.critic_1, args.gamma)

    # with open("saved_models/her_mod_" + args.env_name + "_value" + suffix + ".pkl", 'wb') as f:
    #     pickle.dump(value_estimator, f)
    #     print("Saved value estimator")
    with open("saved_models/her_" + args.env_name + suffix + ".pkl", 'wb') as f:
        pickle.dump(agent.actor_network, f)
        print("Saved agent")

    value_estimator = StateValueEstimator(agent.actor_network, agent.critic.critic_1, args.gamma)

    with open("saved_models/her_" + args.env_name + "_value" + suffix + ".pkl", 'wb') as f:
        pickle.dump(value_estimator, f)
        print("Saved value estimator")