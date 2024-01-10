#!/usr/bin/env python3
"""Module load_frozen_lake."""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Load the pre-made FrozenLakeEnv environment from OpenAI's gym.

    Args:
        desc (list, optional): contains a custom description of the map to
                               load. Defaults to None.
        map_name (str, optional): contains the pre-made map to load. Defaults
                                  to None.
        is_slippery (bool, optional): determines if the ice is slippery.
                                      Defaults to False.
    Returns:
        environment
    """
    env = gym.make('FrozenLake-v1', desc=desc,
                   map_name=map_name, is_slippery=is_slippery)
    return env
