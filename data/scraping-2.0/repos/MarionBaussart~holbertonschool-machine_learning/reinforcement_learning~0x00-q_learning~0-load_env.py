#!/usr/bin/env python3
"""
module containing function load_frozen_lake
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    function that loads the pre-made FrozenLakeEnv evnironment
        from OpenAI's gym
    Args:
        desc: either None or a list of lists
            containing a custom description of the map to load for the
            environment
        map_name: either None or a string containing the pre-made map to load
        is_slippery: boolean to determine if the ice is slippery
    Return: environment
    """
    environment = gym.make('FrozenLake-v1',
                           desc=desc,
                           map_name=map_name,
                           is_slippery=is_slippery)

    return environment
