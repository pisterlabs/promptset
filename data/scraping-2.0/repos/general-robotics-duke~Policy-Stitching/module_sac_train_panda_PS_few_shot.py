import numpy as np
import gym
import my_panda_gym
import os, sys
from sac_arguments import get_args
from mpi4py import MPI
from rl_modules.module_sac_panda_agent_PS_few_shot import module_sac_panda_agent_PS_few_shot
import random
import torch

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
This is for ddpg training now 
"""
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            'joins': 8,
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    # create the sac_agent
    print(MPI.COMM_WORLD.Get_rank())
    env = gym.make(args.env_name, control_type=args.control_type)
    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.device != 'cpu':
        torch.cuda.manual_seed_all(args.seed + MPI.COMM_WORLD.Get_rank())
        # torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # get the environment parameters
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment
    ee_dim = 0
    joint_dim = 0
    if args.env_name == "PandaReach-v2" or args.env_name == "PandaPush-v2" or args.env_name == "PandaSlide-v2" \
            or args.env_name == "PandaSlide-v1" or args.env_name == "PandaPush-v3" or args.env_name == "PandaSlide-v3" \
            or args.env_name == "PandaReach-v3" or args.env_name == "PandaPush-v1" or args.env_name == "PandaPushRocks-v1" \
            or args.env_name == "PandaL5Push-v2" or args.env_name == "PandaL5Push-v3" \
            or args.env_name == "PandaL5PushRocks-v1" or args.env_name == "PandaL3Push-v2" \
            or args.env_name == "PandaL3Push-v3" or args.env_name == "PandaL3PushRocks-v1":
        ee_dim = 6
        joint_dim = 8
    elif args.env_name == "PandaPickAndPlace-v2" or args.env_name == "PandaPush-v4" \
                or args.env_name == "PandaPickAndPlace-v3" or args.env_name == "PandaPickAndPlace-v4" \
            or args.env_name == "PandaL5Push-v4" or args.env_name == "PandaL5PickAndPlace-v2" \
            or args.env_name == "PandaL5PickAndPlace-v3" or args.env_name == "PandaL3Push-v4" \
            or args.env_name == "PandaL3PickAndPlace-v2" or args.env_name == "PandaL3PickAndPlace-v3":
        ee_dim = 7
        joint_dim = 8
    else:
        print("env name wrong in main!")
    print(env.observation_space['observation'].shape[0])
    print(env.observation_space['observation'].shape[0] - ee_dim + env.observation_space['desired_goal'].shape[0])
    print(args.save_data)
    print(args.anchor_mode)

    print("kmeans_real_mixpush2push3_norm anchor mode")
    ta_model_path = 'saved_models9/' + args.ta_env_name + '/task_hidden_dim256-interface_dim128-robot_hidden_dim256-joints_control-seed' + str(
        101) + '-module_sac_real_mixpush2push3_norm_full_model.pt'
    ro_model_path = 'saved_models9/' + args.ro_env_name + '/task_hidden_dim256-interface_dim128-robot_hidden_dim256-joints_control-seed' + str(
        101) + '-module_sac_real_mixpush2push3_norm_full_model.pt'
    sac_trainer = module_sac_panda_agent_PS_few_shot(
        env.observation_space['observation'].shape[0] - ee_dim + env.observation_space['desired_goal'].shape[0],
        joint_dim, args, env, env_params, ta_model_path, ro_model_path)

    sac_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    print("before wierd")
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    print("before launch")
    launch(args)
