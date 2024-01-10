#!/usr/bin/env python3
"""
Module for loading the Frozen Lake envoroment from gym
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Function that loads the pre-made FrozenLakeEnv from OpenAI's gym
    Args:
        desc: Either None or list of lists containing a custom description of
            the map to load for the environment
        man_name: either None or string containing the pre-made map to load
        *** If both desc and map_name are None, the environment will load a
            randomly generated 8x8 map ***
        is_slippery: boolean to determine if the ice is slippery

    Returns: the environment
    """
    environment = gym.make('FrozenLake-v0',
                           desc=desc,
                           map_name=map_name,
                           is_slippery=is_slippery)
    return environment


if __name__ == "__main__":
    import numpy as np
    np.random.seed(0)
    env = load_frozen_lake()
    print(env.desc)
    print(env.P[0][0])
    env = load_frozen_lake(is_slippery=True)
    print(env.desc)
    print(env.P[0][0])
    desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
    env = load_frozen_lake(desc=desc)
    print(env.desc)
    env = load_frozen_lake(map_name='4x4')
    print(env.desc)
