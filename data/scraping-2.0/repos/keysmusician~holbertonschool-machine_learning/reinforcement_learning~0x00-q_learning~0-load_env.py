#!/usr/bin/env python3
""" Defines `load_frozen_lake`. """
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the `FrozenLakeEnv` environment from OpenAI's `gym`.

    desc: Either `None` or a list of lists containing a custom description of
        the map to load for the environment.
    map_name: Either `None` or a string containing the pre-made map to load.
    is_slippery: A boolean to determine if the ice is slippery.

    Returns: The `FrozenLakeEnv` environment.
    """
    if desc is None and map_name is None:
        map_name="8x8"
    return gym.make(
        'FrozenLake-v1', desc=desc, map_name=map_name, is_slippery=is_slippery)
