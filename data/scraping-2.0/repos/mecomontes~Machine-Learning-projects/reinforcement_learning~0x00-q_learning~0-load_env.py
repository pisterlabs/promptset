#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 08:00:42 2021

@author: Robinson Montes
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    loads the pre-made FrozenLakeEnv environment from OpenAI’s gym
    :param desc: either None or a list of lists containing a custom description
        of the map to load for the environment
    :param map_name: either None or a string containing the
        pre-made map to load
    :param is_slippery: boolean to determine if the ice is slippery
    :return: the environment
    """
    FrozenLakeEnv = gym.make('FrozenLake-v0',
                             desc=desc,
                             map_name=map_name,
                             is_slippery=is_slippery)

    return FrozenLakeEnv
