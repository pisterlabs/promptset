#!/usr/bin/env python3
"""
0-load_env
"""
import gym
import numpy as np


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Function that loads the pre-made FrozenLakeEnv
    evnironment from OpenAI’s gym
    """
    env = gym.make('FrozenLake-v0',
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)
    return env