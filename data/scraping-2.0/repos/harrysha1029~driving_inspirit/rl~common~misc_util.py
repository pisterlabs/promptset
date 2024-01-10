# Inspired from OpenAI Baselines

import random

import gym
import numpy as np


def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
