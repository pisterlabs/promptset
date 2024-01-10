#!/usr/bin/env python3
"""loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym:"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym:"""
    env = gym.make('FrozenLake-v0',
                   desc=desc, map_name=map_name,
                   is_slippery=is_slippery)
    return env
