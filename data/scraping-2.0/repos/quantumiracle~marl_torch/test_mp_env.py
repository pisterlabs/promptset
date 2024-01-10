"""
Testing script for vectorized environment wrappers. 

Two types of wrappers are tested:
1. from tianshou library;
2. from openai baseline library.

Environments for test:
1. Openai Gym
2. PettingZoo
"""
# from pettingzoo.butterfly import knights_archers_zombies_v7
# env = knights_archers_zombies_v7.env()
from pettingzoo.atari import boxing_v1
from utils.wrappers import PettingZooWrapper, make_env
import supersuit
import gym
import numpy as np

def test_gym(task, VectorEnv):
    """ 
    Test env parallel DummyVectorEnv (no multiprocess) & SubprocVectorEnv (multiprocess) for single agent gym games.
    """
    env_num = 2
    envs = VectorEnv([lambda: gym.make(task) for _ in range(env_num)])
    assert len(envs) == env_num
    # envs.seed(2)  # which is equal to the next line
    envs.seed(np.random.randint(1000, size=env_num).tolist())
    # envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
    obs = envs.reset()  # reset all environments
    # obs = envs.reset([0, 5, 7])  # reset 3 specific environments
    for i in range(100):
        obs, rew, done, info = envs.step([1] * env_num)  # step synchronously
        envs.render()  # render all environments
    envs.close()  # close all environments

def test_marl(task, VectorEnv, obs_type='ram'):
    """ 
    Test env parallel DummyVectorEnv (no multiprocess) & SubprocVectorEnv (multiprocess) for multi-agent pettingzoo games.
    Use EnvVec Wrappers from Tianshou.
    """
    # env = eval(task).parallel_env(obs_type=obs_type)
    env_num = 2
    envs = VectorEnv([lambda: make_env(task, obs_type=obs_type) for _ in range(env_num)])
    print(envs.action_space)

    assert len(envs) == env_num
    # envs.seed(2)  # which is equal to the next line
    envs.seed(np.random.randint(1000, size=env_num).tolist())
    # envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
    obs = envs.reset()  # reset all environments
    # obs = envs.reset([0, 5, 7])  # reset 3 specific environments
    for i in range(30000):
        print(i)
        actions = [{'first_0':np.random.randint(18), 'second_0':np.random.randint(18)} for i in range(env_num)]
        obs, r, done, info = envs.step(actions)  # step synchronously
        envs.render()  # render all environments
        print(r)
    envs.close()  # close all environments

def test_marl_baseline(task, VectorEnv, obs_type='ram'):
    """ 
    Test env parallel DummyVectorEnv (no multiprocess) & SubprocVectorEnv (multiprocess) for multi-agent pettingzoo games.
    Use EnvVec Wrappers from stable-baseline.
    """
    # env = eval(task).parallel_env(obs_type=obs_type)
    env_num = 2
    envs = VectorEnv([lambda: make_env(task, obs_type=obs_type) for _ in range(env_num)])
    envs.seed(2)  # which is equal to the next line
    obs = envs.reset()  # reset all environments
    for i in range(30000):
        print(i)
        actions = [{'first_0':np.random.randint(18), 'second_0':np.random.randint(18)} for i in range(env_num)]
        obs, r, done, info = envs.step(actions)  # step synchronously
        # envs.render()  # cannot render for stable-baseline env vec wrappers
    envs.close()  # close all environments

if __name__ == '__main__':
    from utils.env import DummyVectorEnv, SubprocVectorEnv
    from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
    # run_parallel2()
    # run_iterate()
    VecEnvSource  = ['tianshou', 'stable_baseline'][0]
    if VecEnvSource == 'tianshou':
        VectorEnv = [DummyVectorEnv, SubprocVectorEnv][1] # tianshou 
        # test_gym('CartPole-v0', VectorEnv)
        # test_marl('slimevolley_v0', VectorEnv)
        test_marl('pong_v1', VectorEnv, 'ram')
    else:
        VectorEnv = [DummyVecEnv, SubprocVecEnv][1] # stable-baseline
        test_marl_baseline('pong_v1', VectorEnv, 'ram')


