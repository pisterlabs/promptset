#!/usr/bin/env python3
"""
Module for the function(s)
    load_frozen_lake(desc=None, map_name=None, is_slippery=False)
"""

import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym.

    Args.
        desc = None or a list of lists containing a custom description of the
               map to load for the environment.
        map_name = None or a string containing the premade map to load.

    Returns.
        The environment.
    """

    env = gym.make(
                    "FrozenLake-v0",
                    desc=desc,
                    map_name=map_name,
                    is_slippery=is_slippery
                  )

    return env
