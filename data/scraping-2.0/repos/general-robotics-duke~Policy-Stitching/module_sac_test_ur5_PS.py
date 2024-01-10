import numpy as np
import gym
import os, sys
from sac_arguments import get_args
from mpi4py import MPI
from rl_modules.module_sac_ur5_agent_PS import module_sac_ur5_agent_PS
import random
import torch
from robot_env.utilities import YCBModels, Camera
from robot_env.robot import UR5Robotiq85
from robot_env.ur5push1 import Ur5Push1
from robot_env.ur5push2 import Ur5Push2
from robot_env.ur5push3 import Ur5Push3
from robot_env.ur5push4 import Ur5Push4

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
This is for ddpg training now 
"""
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_shape,
            'action_max': env.action_space_high,
            'joins': 7,  # Ur5 have 7 joints, 6 on arms 1 for gripper
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    # create the sac_agent
    print(MPI.COMM_WORLD.Get_rank())
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = None
    robot = UR5Robotiq85(pos=(-0.7, -0.1095, 0.1), ori=(0, 0, 0))
    if args.env_name == "Ur5Push1":
        env = Ur5Push1(robot, ycb_models, camera, vis=False)
    elif args.env_name == "Ur5Push2":
        env = Ur5Push2(robot, ycb_models, camera, vis=True)
    elif args.env_name == "Ur5Push3":
        env = Ur5Push3(robot, ycb_models, camera, vis=True)
    elif args.env_name == "Ur5Push4":
        env = Ur5Push4(robot, ycb_models, camera, vis=True)
    else:
        print("wrong environment!")

    # set random seeds for reproduce
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.device != 'cpu':
        torch.cuda.manual_seed_all(args.seed + MPI.COMM_WORLD.Get_rank())
        # torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # get the environment parameters
    env_params = get_env_params(env)

    print("------------")
    print(env.task_input_shape)
    print("------------")
    sac_trainer = module_sac_ur5_agent_PS(env.task_input_shape, env_params['joins'], args, env, env_params)

    task_path = 'saved_models/' + args.ta_env_name + '/task_hidden_dim256-interface_dim128-robot_hidden_dim256-joint_control-seed' + str(
        args.ta_seed) + '-module_sac_push14_norm_full_model9.pt'
    robot_path = 'saved_models/' + args.ro_env_name + '/task_hidden_dim256-interface_dim128-robot_hidden_dim256-joint_control-seed' + str(
        args.ro_seed) + '-module_sac_push14_norm_full_model9.pt'

    sac_trainer.permutation_visualize_3s(task_path, robot_path)

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
