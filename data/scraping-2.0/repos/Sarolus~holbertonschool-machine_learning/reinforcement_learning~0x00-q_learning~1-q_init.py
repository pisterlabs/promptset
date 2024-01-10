#!/usr/bin/env python3
"""
    Script that loads the pre-made FrozenLakeEnv environment from OpenAI’s gym
"""

import numpy as np


def q_init(env):
    """
        Initializes the Q-table

        Args:
            env: is the FrozenLakeEnv instance

        Returns:
            the Q-table as a numpy.ndarray of zeros
    """

    return np.zeros([env.observation_space.n, env.action_space.n])
