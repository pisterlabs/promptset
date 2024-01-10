import argparse
import logging
import os
import sys

import gym
import gym_pull

import ppaquette_gym_doom
from ppaquette_gym_doom.wrappers import SetResolution, ToDiscrete


# The world's simplest agent!
# Taken from OpenAI Gym's random_agent

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

    def learn(self, batch, learning_rate):
        return
