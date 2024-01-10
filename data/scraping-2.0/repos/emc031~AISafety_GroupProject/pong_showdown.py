
"""
Script to run Pong-Atari2600 between two agents.

Also has a couple of baseline agents for illustration, a random agent and an OpenAI baseline agent.

<player1/2> must define class Agent with attribute .showdown_step():
   showdown_step(state = Pong-Atari2600 observation,
                 reward = scalar float)
   returns action = Pong-Atari2600 action, which is a
                    binary list of length 8, e.g.
                    [0,1,0,0,1,0,1,1]
"""

import time
from copy import deepcopy

import numpy as np
import retro

from utils import mirror_observation
from openai_baseline import load_baseline

#import <player1>
#import <player2>

def main():
    """ does thing """

    #player1 = <player1>.Agent()
    #player2 = <player2>.Agent()

    player1 = TrainedBaselineAgent()
    player2 = RandomBaselineAgent()

    # setting up enviroment
    env = retro.make(game='Pong-Atari2600', players=2)
    observation = env.reset()
    done = False
    reward_player1, reward_player2 = (0, 0)

    # game loop
    while not done:

        action_player1 = player1.showdown_step(observation,
                                               reward_player1)
        action_player2 = player2.showdown_step(mirror_observation(observation),
                                               reward_player2)

        observation, reward, done, info = env.step(
            np.concatenate((action_player1, action_player2))
        )

        reward_player1, reward_player2 = tuple(reward)

        env.render()
        time.sleep(0.015)

        if done:
            obs = env.reset()
            env.close()
            print('Final score: '+str(info))
##


class RandomBaselineAgent:
    """ an agent to play pong that always just does random actions """

    def showdown_step(self,
                      observation,
                      reward):
        """
        args:
        observation = matrix of pixels from pong (ignored)
        reward = float (ignored)
        returns:
        action = binary array of length 8
        """
        return np.random.randint(0, 2, 8)
##


class TrainedBaselineAgent:
    """ here's one I made earlier """

    def __init__(self):
        self.model = load_baseline()
    ##

    def showdown_step(self,
                      observation,
                      reward):
        """
        args:
        observation = matrix of pixels from pong
        reward = float (ignored)
        returns:
        action = binary array of length 8
        """
        actions, values, observation, _ = self.model.step(observation)
        return actions[0]
    ##


if __name__ == "__main__":
    main()
