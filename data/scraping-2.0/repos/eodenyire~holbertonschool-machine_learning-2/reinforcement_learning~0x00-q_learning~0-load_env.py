#!/usr/bin/env python3
"""
Module used to
"""

import gym
import numpy as np
import random


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """[Function that loads the pre-made FrozenLakeEnv env from OpenAI’s gym]

    Args:
        desc ([list], optional):    [list of lists with a custom description
                                    of the map to load for the environment].
                                    Defaults to None.
        map_name ([str], optional): [string with the pre-made map to load].
                                Defaults to None.
        is_slippery (bool, optional): [description]. Defaults to False.
    Returns: the environment
    """

    # load all enviroments
    all_envs = gym.envs.registry.all()

    # load the very basic taxi environment.
    # env = gym.make("Taxi-v2")

    enviroment_name = 'FrozenLake-v0'
    env = gym.make(enviroment_name,
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)
    # To initialize the environment, we must reset it.
    env.reset()

    # determine the total number of possible states:
    # env.observation_space.n

    # If you would like to visualize the current state, type the following:
    # env.render()

    return env
