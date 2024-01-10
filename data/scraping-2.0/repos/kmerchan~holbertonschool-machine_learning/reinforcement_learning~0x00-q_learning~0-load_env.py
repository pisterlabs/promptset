#!/usr/bin/env python3
"""
Defines function that loads pre-made FrozenLakeEnv environment
from OpenAI's gym
"""


import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads pre-made FrozenLakeEnv environment from OpenAI's gym

    returns:
        the environment
    """
    env = gym.make("FrozenLake-v0",
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)
    return env
