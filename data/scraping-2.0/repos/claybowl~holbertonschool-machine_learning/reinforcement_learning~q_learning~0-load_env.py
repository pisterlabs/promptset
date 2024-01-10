#!/usr/bin/env python3
"""Module 0-load_env
loads the pre-made FrozenLakeEnv environment
from OpenAIs gym.
"""
import numpy as np
import gymnasium as gym
import random
import time
from IPython.display import clear_output
from gym.envs.toy_text.frozen_lake import generate_random_map


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Load the FrozenLake environment from OpenAI's gym.

    Parameters:
    desc -- None or a list of lists containing a custom
    description of the map to load for the environment
    map_name -- None or a string containing the pre-made map to load
    is_slippery -- a boolean to determine if the ice is slippery

    If both desc and map_name are None, the environment will
    load a randomly generated 8x8 map.

    Returns:
    The FrozenLake environment.
    """
    if desc is None and map_name is None:
        desc = generate_random_map(size=8)

    return gym.make('FrozenLake-v0', desc=desc,
                    map_name=map_name, is_slippery=is_slippery)
