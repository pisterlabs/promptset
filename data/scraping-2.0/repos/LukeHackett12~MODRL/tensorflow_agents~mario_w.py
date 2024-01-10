from tensorboard.plugins.hparams import api as hp
from tensorflow import keras, Tensor
import tensorflow as tf
import datetime
import random
from collections import namedtuple, deque
from typing import NamedTuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.interpolate.interpolate import interp1d
from enum import Enum

from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from gym.spaces import Box
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import argparse
import math
from copy import deepcopy
from replay_buffer_policy import PrioritizedReplayBuffer
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoint", required=False)
parser.add_argument("-e", "--episode", required=False)
parser.add_argument("-x", "--epsilon", required=False)

# %matplotlib inline
GROUP_NUM = 20

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

class PolEnum(Enum):
    xpos = 0
    time = 1
    coin = 2
    deaths = 3
    random = 4


class Transition(NamedTuple):
    currStates: Tensor
    actions: Tensor
    rewards: Tensor
    nextStates: Tensor
    dones: Tensor


class DQNAgent:
    def __init__(self, stateShape, actionSpace, numPicks, memorySize, numRewards, sync=1000, burnin=10000, alpha=0.0001, epsilon=1, epsilon_decay=100000, epsilon_min=0.01, gamma=0.95, checkpoint=None):
        self.numPicks = numPicks
        self.stateShape = stateShape
        self.actionSpace = actionSpace
        self.numRewards = numRewards

        self.step = 0

        self.sync = sync
        self.burnin = burnin
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.walpha = 0.01
        self.delay = 1.1

        self.train_network = self.createNetwork(
            stateShape, (actionSpace.n), self.alpha
        )
        self.target_network = self.createNetwork(
            stateShape, (actionSpace.n), self.alpha
        )

        # Store network weights for each policy
        self.policy_train_weights = [
            deepcopy(self.train_network.get_weights())
        ] * self.numRewards
        self.policy_target_weights = [
            deepcopy(self.train_network.get_weights())
        ] * self.numRewards

        # Individual replay buffers for policies and for w net
        self.replayMemory = []
        for i in range(self.numRewards):
            self.replayMemory.append(PrioritizedReplayBuffer(memorySize, 0.6))

        # Create and store network weights for W-values
        self.w_train_network = self.createNetwork(stateShape, numRewards, self.alpha)
        self.wnet_train_weights = [
            deepcopy(self.w_train_network.get_weights())
        ] * self.numRewards

    def createNetwork(self, n_input, n_output, learningRate):
        model = keras.models.Sequential()

        model.add(keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=n_input))
        model.add(keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', data_format="channels_first"))
        model.add(keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu', data_format="channels_first"))
        model.add(keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', data_format="channels_first"))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='linear'))
        model.add(keras.layers.Dense(n_output, activation='linear'))

        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(lr=learningRate))
        return model

    def trainDQN(self):
        if len(self.replayMemory[0]) <= self.numPicks or len(self.replayMemory[0]) <= self.burnin:
            return [(0, 0)] * self.numRewards

        agentsLoss = []
        beta = 0.4 + self.step * (1.0 - 0.4) / 1000000

        for i in range(self.numRewards):
            samples = self.replayMemory[i].sample(self.numPicks, beta)
            (
                currStates,
                actions,
                policies,
                rewards,
                nextStates,
                dones,
                weights,
                indices,
            ) = samples

            currStates = np.array(currStates)
            nextStates = np.array(nextStates)

            rewards = (
                np.array(rewards)
                .reshape(
                    self.numPicks,
                )
                .astype(float)
            )
            actions = (
                np.array(actions)
                .reshape(
                    self.numPicks,
                )
                .astype(int)
            )
            policies = (
                np.array(policies)
                .reshape(
                    self.numPicks,
                )
                .astype(int)
            )

            dones = np.array(dones).astype(bool)
            notDones = (~dones).astype(float)
            dones = dones.astype(float)

            self.train_network.set_weights(self.policy_train_weights[i])
            Q_currents_all = self.train_network(currStates, training=False).numpy()

            self.target_network.set_weights(self.policy_target_weights[i])
            Q_futures_all = (
                self.target_network(nextStates, training=False).numpy().max(axis=1)
            )

            Q_currents = np.copy(Q_currents_all)
            Q_futures = np.copy(Q_futures_all)

            # Q-Learning
            Q_currents[np.arange(self.numPicks), actions] = (
                rewards * dones + (rewards + Q_futures * self.gamma) * notDones
            )
            lossQ = self.train_network.train_on_batch(currStates, Q_currents)
            self.policy_train_weights[i] = deepcopy(self.train_network.get_weights())

            # PER Update
            prios = (np.abs(lossQ) * weights) + 1e-5
            self.replayMemory[i].update_priorities(indices, prios)

            lossW = 0

            # Leave in exploration actions for now, can remove with "policy[p] != -1"
            inverted_policy_mask = np.array(
                [p for p in range(self.numPicks) if policies[p] != i]
            )
            if len(inverted_policy_mask) > 0:
                # W-Learning
                self.w_train_network.set_weights(self.wnet_train_weights[i])

                currStatesNP = currStates[inverted_policy_mask]
                policiesNP = policies[inverted_policy_mask]
                rewardNP = rewards[inverted_policy_mask]
                donesNP = dones[inverted_policy_mask]
                notDonesNP = notDones[inverted_policy_mask]

                Q_currents_np = Q_currents_all[inverted_policy_mask].max(axis=1)
                Q_futures_np = Q_futures_all[inverted_policy_mask]

                w_train = self.w_train_network(currStatesNP, training=False).numpy()

                # maybe (Q_currents_not_policy - ((rewardNP * dones) + (self.gamma * Q_futures_not_policy) * notDonesNP)) * walpha^delay ?
                w_train[np.arange(len(inverted_policy_mask)), policiesNP] = (
                    (1 - self.walpha)
                    * w_train[np.arange(len(inverted_policy_mask)), policiesNP]
                ) + (
                    (self.walpha ** self.delay)
                    * (
                        Q_currents_np
                        - (
                            (rewardNP * donesNP)
                            + (self.gamma * Q_futures_np) * notDonesNP
                        )
                    )
                )
                lossW = self.w_train_network.train_on_batch(currStatesNP, w_train)
                self.wnet_train_weights[i] = self.w_train_network.get_weights()

            agentsLoss.append((lossQ, lossW))

        return agentsLoss

    def selectAction(self, state):
        self.step += 1
        state = np.expand_dims(np.array(state), 0)

        if self.step % self.sync == 0:
            self.policy_target_weights = deepcopy(self.policy_train_weights)

        emptyPolicies = [0] * self.numRewards
        policy, qs, ws = (-1, -1, emptyPolicies)
        random = True
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            ws = []
            if np.random.rand(1) < self.epsilon:
                policy = np.random.randint(0, self.numRewards)
            else:
                for i in range(self.numRewards):
                    self.w_train_network.set_weights(self.wnet_train_weights[i])
                    w_val = self.w_train_network(state, training=False).numpy()[0]
                    ws.append(w_val[np.argmax(w_val)])
                    random = False

                policy = np.argmax(ws)

            self.train_network.set_weights(self.policy_train_weights[policy])
            pred = np.squeeze(self.train_network(state, training=False).numpy(), axis=0)
            action = np.argmax(pred)
            qs = np.max(pred)

        return action, policy, qs, ws, random

    def addMemory(self, state, action, policy, reward, nextState, done):
        for i in range(self.numRewards):
            self.replayMemory[i].add(state, action, policy, reward[i], nextState, done)

    def save(self, ep):
        save_path = (
            f"./mario_w_{int(ep)}.chkpt"
        )
        weights = []
        for i in range(self.numRewards):
            train_w = self.policy_train_weights[i]
            target_w = self.policy_train_weights[i]
            w_w = self.wnet_train_weights[i]

            weights.append([train_w, target_w, w_w])

        with open(save_path, "wb") as f:
            pickle.dump(weights, f)
        
        print(f"MarioNet saved to {save_path} done!")


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

# Taken from OpenAI baselines: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MarioBaseline(object):
    def __init__(self, episodes, checkpoint, current_episode, epsilon):
        self.current_episode = current_episode
        self.episodes = episodes

        self.episode_score = []
        self.episode_qs = []
        self.episode_distance = []
        self.episode_loss = []
        self.episode_policies = []

        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 4))
        self.fig.canvas.draw()

        self.env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        # Apply Observation Wrappers
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, 84)
        # Apply Control Wrappers
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.env = NoopResetEnv(self.env)
        # Apply Frame Wrappers
        self.env = SkipFrame(self.env, 4)
        self.env = FrameStack(self.env, 4)

        self.agent = DQNAgent(stateShape=(4, 84, 84), actionSpace=self.env.action_space, numPicks=32, memorySize=20000, numRewards=4, epsilon=epsilon, checkpoint=checkpoint)

    def train(self):
        for _ in range(self.episodes):
            self.episode()
            self.current_episode += 1


        self.env.close()

    def episode(self):
        done = False
        rewardsSum = 0
        qSum = 0
        qActions = 1
        lossSum = 0
        policies = [0] * (4 + 1)
        lossSums = [0] * (4)

        state = np.array(self.env.reset())
        maxDistance = -1000000
        lastX = 0
        lastT = 0
        lastC = 0

        while not done:
            action, policy, qs, ws, random = self.agent.selectAction(state)
            policies[policy] += 1
            obs, _, done, info = self.env.step(action)
            #self.env.render()
            

            if info['x_pos'] > maxDistance:
                maxDistance = info['x_pos']
            rewardX = info['x_pos'] - lastX
            lastX = info['x_pos']
            rewardT = info['time'] - lastT
            if rewardT > 0: rewardT = 0
            lastT = info['time']
            rewardC = info['coins'] - lastC
            lastC = info['coins']
            rewardD = self.env.unwrapped._death_penalty

            next_state = np.array(obs)
            rewardsSum = np.add(rewardsSum, rewardX)
            rewardsSum = np.add(rewardsSum, rewardT)
            rewardsSum = np.add(rewardsSum, rewardC)
            rewardsSum = np.add(rewardsSum, rewardD)

            self.agent.addMemory(state, action, policy, [rewardX, rewardT, rewardC, rewardD], next_state, done)
            loss = self.agent.trainDQN()
            state = next_state
            lossSums = [lossSums[i] + loss[i][0] for i in range(len(lossSums))]
        
        self.agent.epsilon = self.agent.epsilon_min + (1 - self.agent.epsilon_min) * math.exp(-1 * ((self.agent.step + 1) / self.agent.epsilon_decay))

        print("now epsilon is {}, the reward is {} with loss {} in episode {}, step {}, dist {}".format(
            self.agent.epsilon, rewardsSum, lossSums, self.current_episode, self.agent.step, maxDistance))

        self.episode_score.append(rewardsSum)
        self.episode_policies.append(policies)

        if self.current_episode % 200 == 0:
            self.agent.save(self.current_episode)
            self.plot()

    def plot(self):
        spline_x = np.linspace(0, self.current_episode, num=self.current_episode)

        ep_scores = np.array(self.episode_score)
        ep_groups = [
            ep_scores[i * GROUP_NUM : (i + 1) * GROUP_NUM]
            for i in range((len(ep_scores) + GROUP_NUM - 1) // GROUP_NUM)
        ]
        # Pad for weird numpy error for now
        ep_groups[-1] = np.append(
            ep_groups[-1], [np.mean(ep_groups[-1])] * (GROUP_NUM - len(ep_groups[-1]))
        )
        x_groups = [i * GROUP_NUM for i in range(len(ep_groups))]

        self.ax[0].clear()
        if len(x_groups) > 5:
            ep_avgs = np.mean(ep_groups, 1)
            avg_spl = interp1d(
                x_groups, ep_avgs, kind="cubic", fill_value="extrapolate"
            )
            ep_std = np.std(ep_groups, 1)
            std_spl = interp1d(x_groups, ep_std, kind="cubic", fill_value="extrapolate")
            self.ax[0].plot(spline_x, avg_spl(spline_x), lw=0.7, c="blue")
            self.ax[0].fill_between(
                spline_x,
                avg_spl(spline_x) - std_spl(spline_x),
                avg_spl(spline_x) + std_spl(spline_x),
                alpha=0.5,
                facecolor="red",
                interpolate=True,
            )

        self.ax[0].title.set_text("Training Score")
        self.ax[0].set_xlabel("Episode")
        self.ax[0].set_ylabel("Score")

        policies = np.transpose(self.episode_policies)
        colors = pl.cm.jet(np.linspace(0, 1, len(policies) * 2))

        self.ax[1].clear()
        self.ax[1].title.set_text("Policy Choices")
        for i, policy in enumerate(policies):
            if len(x_groups) > 5:
                ep_groups = [
                    policy[i * GROUP_NUM : (i + 1) * GROUP_NUM]
                    for i in range((len(policy) + GROUP_NUM - 1) // GROUP_NUM)
                ]
                # Pad for weird numpy error for now
                ep_groups[-1] = np.append(
                    ep_groups[-1],
                    [np.mean(ep_groups[-1])] * (GROUP_NUM - len(ep_groups[-1])),
                )
                x_groups = [i * GROUP_NUM for i in range(len(ep_groups))]

                ep_avgs = np.mean(ep_groups, 1)
                avg_spl = interp1d(
                    x_groups, ep_avgs, kind="cubic", fill_value="extrapolate"
                )
                ep_std = np.std(ep_groups, 1)
                std_spl = interp1d(
                    x_groups, ep_std, kind="cubic", fill_value="extrapolate"
                )
                self.ax[1].plot(
                    spline_x,
                    avg_spl(spline_x),
                    lw=0.7,
                    c=colors[i],
                    label="{} policy".format(PolEnum(i).name),
                )
                self.ax[1].fill_between(
                    spline_x,
                    avg_spl(spline_x) - std_spl(spline_x),
                    avg_spl(spline_x) + std_spl(spline_x),
                    alpha=0.5,
                    facecolor=colors[-1 - i],
                    interpolate=True,
                )

        self.ax[1].legend()

        self.fig.canvas.draw()
        plt.savefig("mario_w_pddqn_{}.png".format(self.current_episode))


args = parser.parse_args()
checkpoint = args.checkpoint
current_episode = args.episode
epsilon = args.epsilon

if current_episode == None:
    current_episode = 0
else:
    current_episode = int(current_episode)
if epsilon == None:
    epsilon = 1
else:
    epsilon = float(epsilon)

agent = MarioBaseline(10000, checkpoint, current_episode, epsilon)
agent.train()

