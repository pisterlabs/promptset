import sys
import os
o_path = os.getcwd() # 返回当前工作目录
sys.path.append(o_path) # 添加自己指定的搜索路径

import numpy as np
import gym
import os, sys
from arguments import get_args
import torch
from mpi4py import MPI
from subprocess import CalledProcessError

import time
from spinup_utils.logx import setup_logger_kwargs, colorize
from spinup_utils.logx import EpochLogger
from spinup_utils.print_logger import Logger

from gym_myrobot.envs.robot_reach import ReachEnv
from gym_myrobot.envs.real_arm_env import RealarmEnv

import matplotlib.pyplot as plt
"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
but I ignore it~

Goal环境，网络的好坏取决于success rate， 也就是成功率，而不取决于每个回合的奖励高低，只要回合内部有一步成功了，那就算是成功了！

"""

def test(net, env, args):
    rs = []
    env_target = []
    for i in range(5):
        episode_r = 0
        obs = env.reset()
        env_target.append(obs['desired_goal'])
        observation = obs['observation']
        goal = obs['desired_goal']
        observation = np.concatenate((observation, goal))
        # print(observation.shape)
        for j in range(200):
            action = net.get_action(observation)
            obs, r, d, _ = env.step(action)
            episode_r += r
            observation = obs['observation']
            goal = obs['desired_goal']
            observation = np.concatenate((observation, goal))
            if d:
                break
        rs.append(episode_r)
    plt.figure()
    plt.plot(rs)
    plt.show()
    print(env_target)


def launch(net, args):
    # env = gym.make(args.env_name)
    env = ReachEnv(usegui=True)  
    # 为了保证两个环境中产生的随机目标点是一致的，需要申明seed
    seed = args.seed
    # seed = 1

    env.seed(seed)
    np.random.seed(seed)

    envreal = RealarmEnv(reward_type='sparse',use_gripper=True)
    # 为了保证两个环境中产生的随机目标点是一致的，需要申明seed, 从实验结果来看，的确是一致的随机数
    envreal.seed(seed)
    np.random.seed(seed)

    try:
        s_dim = env.observation_space.shape[0]
    except:
        s_dim = env.observation_space.spaces['observation'].shape[0] + \
                env.observation_space.spaces['desired_goal'].shape[0]

    act_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    """        
        torch1.17.1，gpu_id: 1 device: cuda:0，用的是物理上的0卡；
        cuda的序号仍然是按照物理序号；
        torch1.3.1，gpu_id: 1 device: cuda:0，用的是物理上的1卡，
        torch1.3.1，gpu_id: 1 device: cuda:1，报错：invalid device ordinal；
        torch1.3.1，gpu_id: 1,3 device: cuda:1，用的是物理上的3卡，
        有点类似于指定GPU-ID后，cuda会重新排序。        
    """

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() and args.gpu_id != -1 else 'cpu')
    print("gpu_id:", args.gpu_id,
          "device:", device)

    net = net(act_dim=act_dim,
              obs_dim=s_dim,
              a_bound=a_bound,
              per_flag=args.per,
              her_flag=args.her,
              action_l2=args.action_l2,
              state_norm=args.state_norm,
              gamma=args.gamma,
              sess_opt=args.sess_opt,
              seed=args.seed,
              clip_return=args.clip_return,
              device=device,
              )

    net.load_simple_network('/home/zp/deeplearning/open_manipulator_x_hasgravity/DRLib/actor.pth')
    # restore_path = 'HER_DRLib_exps/2021-02-22_HER_TD3Torch_FetchPush-v1/2021-02-22_14-46-52-HER_TD3Torch_FetchPush-v1_s123/actor.pth'
    # net.load_simple_network(restore_path)
    # trainer(net, env, args)
    # test_sim(net, env, args)
    # 测试仿真环境
    test(net, env, args)

    # 测试真实环境
    test(net, envreal, args)

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    
    # get the params
    args = get_args()
    from algos.tf1.td3_sp.TD3_per_her import TD3
    from algos.tf1.ddpg_sp.DDPG_per_her import DDPG
    from algos.tf1.sac_sp.SAC_per_her import SAC
    from algos.tf1.sac_auto.sac_auto_per_her import SAC_AUTO

    from algos.pytorch.td3_sp.td3_per_her import TD3Torch
    from algos.pytorch.ddpg_sp.ddpg_per_her import DDPGTorch
    from algos.pytorch.sac_sp.sac_per_her import SACTorch

    RL_list = [TD3, DDPG, SAC, SAC_AUTO, TD3Torch, DDPGTorch, SACTorch]

    [launch(net=net, args=args) for net in RL_list if net.__name__ == args.RL_name]
