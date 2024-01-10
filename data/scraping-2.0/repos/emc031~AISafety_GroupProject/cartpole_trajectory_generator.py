
import time
from copy import deepcopy

import numpy as np
import gym
from gym import wrappers

from utils import mirror_observation
from openai_baseline import load_baseline

class CartpoleTragetoryGenerator:
    """ Generates tragetories based on two agents playing pong"""

    def __init__(self, player1, player2, tragetory_length = 80):
        #self._env = gym.make('Pong-v0')
        self._env = gym.make('CartPole-v0')
        self._env = wrappers.Monitor(self._env,
                                     '.',
                                     video_callable=False,
                                     force=True)
        
        self._tragetory_length = tragetory_length
    #

    def build(self):
        observation = self._env.reset()
        done = False

        # game loop
        out = list()
        for _ in range(self._tragetory_length):
            if not done:
                action = self._env.action_space.sample()
                observation, reward, done, info = self._env.step(action)
            ##

            out.append(self._env.render(mode='rgb_array')[::2,::2,:])
        ##
        self._env.close()
        return out
    #
#

def generate_cartpole_tragetory(player1, player2, tragetory_length = 120):
    """ Generate a tragetory from the given pair of pong player agents """
    generator = CartpoleTragetoryGenerator(player1, player2, tragetory_length)
    return generator.build()
#


##

