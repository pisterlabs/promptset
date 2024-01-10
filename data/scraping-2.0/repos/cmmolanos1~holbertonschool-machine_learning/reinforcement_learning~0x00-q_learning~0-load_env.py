#!/usr/bin/env python3

from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads the pre-made FrozenLakeEnv environment from OpenAI’s gym.

    Args:
        desc (list): is either None or a list of lists containing a custom
                     description of the map to load for the environment.
        map_name (str):  is either None or a string containing the pre-made
                         map to load.
        is_slippery (bool): determine if the ice is slippery.

    Returns:
        the environment.
    """

    env = FrozenLakeEnv(desc, map_name, is_slippery)

    return env
