#!/usr/bin/env python3
"""Task 0. Load the Environment"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the FrozenLakeEnv environment from OpenAI Gym"""
    env = gym.make("FrozenLake-v1",
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)
    return env
