#!/usr/bin/env python3
"""
Loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym
    :param desc: is either None or a list of lists containing a custom
    description of the map to load for the environment
    :param map_name: is either None or a string containing the pre-made map
    to load
    :param is_slippery: is a boolean to determine if the ice is slippery
    :return: the environment
    """
    env = gym.make(id='FrozenLake-v0', desc=desc, map_name=map_name,
                   is_slippery=is_slippery)
    return env
