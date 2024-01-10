#!/usr/bin/env python3
"""
Loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gy
"""
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Args:
        desc: list of lists containing a custom description of
        the map to load for the environment
        map_name: string containing the pre-made map to load
        is_slippery: boolean to determine if the ice is slippery
    Returns:
        The environment
    Note:
         If both desc and map_name are None,
         the environment will load a randomly generated 8x8 map
    """
    if desc is None and map_name is None:
        environment = gym.make("FrozenLake-v1",
                               desc=generate_random_map(size=8),
                               map_name='8x8',
                               is_slippery=is_slippery,
                               render_mode="rgb_array")
    else:
        environment = gym.make("FrozenLake-v1",
                               desc=desc,
                               map_name=map_name,
                               is_slippery=is_slippery,
                               render_mode="rgb_array")
    return environment
