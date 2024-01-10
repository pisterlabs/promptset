from directed_exploration.mcts.mcts import MCTS, getBatchMCTSActionProbs
from directed_exploration.mcts.pytorch_classification.utils import AverageMeter
from directed_exploration.mcts.pytorch_classification.utils.progress.progress.bar import Bar
from directed_exploration.utils.tf_util import log_tensorboard_scalar_summaries

import logging
import numpy as np
from collections import deque
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from multiprocessing.pool import ThreadPool

logger = logging.getLogger(__name__)


def discount_with_dones(rewards, dones, gamma):
    # From OpenAI Baselines
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, subproc_env_group, nnet, args, summary_writer, reward_discount_factor=0.99):
        self.subproc_env_group = subproc_env_group
        self.nenvs = subproc_env_group.nenvs
        self.nnet = nnet
        self.args = args
        self.summary_writer = summary_writer
        self.gamma = reward_discount_factor
        self.search_trees = [MCTS(self.nnet, args) for _ in range(self.nenvs)]
        # self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()

        self.observations = subproc_env_group.reset()
        self.dones = [False for _ in range(self.nenvs)]
        self.env_states = self.subproc_env_group.clone_full_states()

        self.thread_pool = ThreadPool(processes=self.nenvs)



        self.running_first_thread_episode_reward = 0
        self.first_thread_episodes_completed = 0

    def run_batch(self, nsteps):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        mb_obs = []
        mb_pi_targets = []
        mb_dones = []
        mb_rewards = []

        latest_value_estimates = None

        for i in range(nsteps):

            mcts_policies, latest_value_estimates = getBatchMCTSActionProbs(
                mcts_instances=self.search_trees,
                states=self.env_states,
                envs=self.subproc_env_group.env_handles,
                observations=self.observations,
                reward_discount_factor=self.gamma,
                thread_pool=self.thread_pool,
                temps=[self.args.temp for _ in range(self.nenvs)]
            )

            mb_obs.append(self.observations)
            mb_pi_targets.append(mcts_policies)

            actions = [np.random.choice(len(pi), p=pi) for pi in mcts_policies]

            self.subproc_env_group.restore_full_states(self.env_states)
            next_observations, next_rewards, next_dones, infos = self.subproc_env_group.step(actions)

            # import cv2
            # from baselines.common.tile_images import tile_images
            # cv2.imshow('obs', np.expand_dims(next_observations[0,:,:,0], axis=2))
            # cv2.waitKey(1)
            # self.subproc_env_group.render()

            self.running_first_thread_episode_reward += next_rewards[0]
            if next_dones[0]:
                log_tensorboard_scalar_summaries(summary_writer=self.summary_writer,
                                                 values=[self.running_first_thread_episode_reward],
                                                 tags=['per_episode_reward'],
                                                 step=self.first_thread_episodes_completed)

                logger.debug("{} first thread episodes completed, had reward of {}"
                             .format(self.first_thread_episodes_completed + 1,
                                     self.running_first_thread_episode_reward))

                self.running_first_thread_episode_reward = 0
                self.first_thread_episodes_completed += 1

            mb_dones.append(next_dones)
            mb_rewards.append(next_rewards)

            self.dones = next_dones
            self.observations = next_observations
            self.env_states = self.subproc_env_group.clone_full_states()

        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        for n, (rewards, dones, value_estimates) in enumerate(zip(mb_rewards, mb_dones, latest_value_estimates)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value_estimates], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)

            mb_rewards[n] = rewards

        mb_pi_targets = np.asarray(mb_pi_targets, dtype=np.float32)

        mb_obs = np.reshape(mb_obs, newshape=(-1, *mb_obs.shape[2:]))
        mb_pi_targets = np.reshape(mb_pi_targets, newshape=(-1, *mb_pi_targets.shape[2:]))
        mb_rewards = np.reshape(mb_rewards, newshape=(-1, *mb_rewards.shape[2:]))

        return mb_obs, mb_pi_targets, mb_rewards

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                # iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                batch_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.num_batches)
                end = time.time()

                for batch_number in range(self.args.num_batches):
                    self.search_trees = [MCTS(self.nnet, self.args) for _ in range(self.nenvs)]
                    scaled_obs_batch, policy_targets, value_targets = self.run_batch(self.args.batch_nsteps)
                    self.nnet.train_on_batch(scaled_obs_batch, policy_targets, value_targets)
                    # bookkeeping + plot progress
                    batch_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = '({batch}/{max_batches}) Batch Time: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        batch=batch_number + 1, max_batches=self.args.num_batches, bt=batch_time.avg,
                        total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

            self.nnet.save_model()
                # save the iteration examples to the history
                # self.trainExamplesHistory.append(iterationTrainExamples)

            # if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
            #     print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
            #           " => remove the oldest trainExamples")
            #     self.trainExamplesHistory.pop(0)
            # # backup history to a file
            # # NB! the examples were collected using the model from the previous iteration, so (i-1)
            # self.saveTrainExamples(i - 1)

            # shuffle examlpes before training
            # trainExamples = []
            # for e in self.trainExamplesHistory:
            #     trainExamples.extend(e)
            # shuffle(trainExamples)
            #

            #
            # print("training")


            # # training new network, keeping a copy of the old one
            # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # pmcts = MCTS(self.game, self.pnet, self.args)
            #
            # self.nnet.train(trainExamples)
            # nmcts = MCTS(self.game, self.nnet, self.args)
            #
            # print('PITTING AGAINST PREVIOUS VERSION')
            # arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
            #               lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            # pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            #
            # print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            # if pwins + nwins > 0 and float(nwins) / (pwins + nwins) < self.args.updateThreshold:
            #     print('REJECTING NEW MODEL')
            #     self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # else:
            #     print('ACCEPTING NEW MODEL')
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
