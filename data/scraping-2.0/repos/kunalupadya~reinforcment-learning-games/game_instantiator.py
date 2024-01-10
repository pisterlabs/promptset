from openaigymexample import init_gym_game, restore_saved_agent, get_trainer
import os

GAMES = {
    "CartPole-v1":False,
    "Pong-v0":True,
    "SpaceInvaders-v0":True,
    "LunarLander-v2":False,
    "MountainCar-v0":False
}

NON_ATARI = {'CartPole-v1', 'LunarLander-v2', 'MountainCar-v0', 'Blackjack-v0', 'BipedalWalker-v3','BipedalWalkerHadcore-v3'}

class GameInstantiator():
    def getAgent(self, chosen_game, n_iter, algorithm, config):
        potential_path = chosen_game + '/' + algorithm + '/checkpoint_' + str(n_iter) + '/checkpoint-' + str(n_iter)
        if os.path.isfile(potential_path):
            return restore_saved_agent(chosen_game, potential_path, algorithm)
        else:
            return init_gym_game(chosen_game, n_iter, algorithm, config)

    def getTrainer(self, algorithm):
        return get_trainer(algorithm)

