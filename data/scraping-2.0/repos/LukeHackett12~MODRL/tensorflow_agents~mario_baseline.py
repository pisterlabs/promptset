from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import NamedTuple
from collections import namedtuple, deque
import random
import datetime
import time

import tensorflow as tf
from tensorflow import keras, Tensor

class Transition(NamedTuple):
    currStates: Tensor
    actions: Tensor
    rewards: Tensor
    nextStates: Tensor
    dones: Tensor


class DQNAgent:
    def __init__(self, stateShape, actionSpace, numPicks, memorySize, sync=100000, burnin=100, alpha=0.0001, epsilon=1, epsilon_decay=0.99999975, epsilon_min=0.01, gamma=0.9):
        self.numPicks = numPicks
        self.replayMemory = deque(maxlen=memorySize)
        self.stateShape = stateShape
        self.actionSpace = actionSpace

        self.step = 0

        self.sync = sync
        self.burnin = burnin
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.trainNetwork = self.createNetwork(
            stateShape, actionSpace.n, self.alpha)
        self.targetNetwork = self.createNetwork(
            stateShape, actionSpace.n, self.alpha)

        self.targetNetwork.set_weights(self.trainNetwork.get_weights())

    def createNetwork(self, n_input, n_output, learningRate):
        model = keras.models.Sequential()

        model.add(keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=n_input))
        model.add(keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='linear'))
        model.add(keras.layers.Dense(n_output, activation='linear'))

        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(lr=learningRate))
        print(model.summary())
        return model

    def trainDQN(self):
        if len(self.replayMemory) <= self.numPicks or len(self.replayMemory) <= self.burnin:
            return 0

        samples = random.sample(self.replayMemory, self.numPicks)
        batch = Transition(*zip(*samples))
        currStates, actions, rewards, nextStates, _ = batch

        currStates = np.squeeze(np.array(currStates))
        Q_currents = self.trainNetwork(currStates, training=False).numpy()

        nextStates = np.squeeze(np.array(nextStates))
        Q_futures = self.targetNetwork(nextStates, training=False).numpy().max(axis=1)

        rewards = np.array(rewards).reshape(self.numPicks,).astype(float)
        actions = np.array(actions).reshape(self.numPicks,).astype(int)

        Q_currents[np.arange(self.numPicks), actions] = rewards + Q_futures * self.gamma
        hist = self.trainNetwork.train_on_batch(currStates, Q_currents)

        return hist.history['loss'][0]

    def selectAction(self, state):
        self.step += 1

        if self.step % self.sync == 0:
            self.targetNetwork.set_weights(self.trainNetwork.get_weights())

        q = -100000
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            preds = np.squeeze(self.trainNetwork(
                state, training=False).numpy(), axis=0)
            action = np.argmax(preds)
            q = preds[action]
        return action, q

    def addMemory(self, memory):
        self.replayMemory.append(memory)

    def save(self, ep):
        save_path = (
            f"./mario_{int(ep)}.chkpt"
        )
        self.trainNetwork.save(save_path)
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
    def __init__(self, episodes):
        self.current_episode = 0
        self.episodes = episodes

        self.episode_score = []
        self.episode_qs = []
        self.episode_distance = []
        self.episode_loss = []

        self.fig, self.ax = plt.subplots(2, 2)
        self.fig.canvas.draw()
        plt.show(block=False)

        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        # Apply Observation Wrappers
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, 84)
        # Apply Control Wrappers
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.env = NoopResetEnv(self.env)
        # Apply Frame Wrappers
        self.env = SkipFrame(self.env, 4)
        self.env = FrameStack(self.env, 4)

        self.agent = DQNAgent(stateShape=(84, 84, 4),
                              actionSpace=self.env.action_space, numPicks=32, memorySize=100000)

    def train(self):
        for _ in range(self.episodes):
            self.episode()
            self.plot()
            self.current_episode += 1

        self.env.close()

    def episode(self):
        done = False
        rewardsSum = 0
        qSum = 0
        qActions = 1
        lossSum = 0

        state = np.array(self.env.reset()).transpose(3, 1, 2, 0)
        maxDistance = -1000000
        lastX = 0

        while not done:
            action, q = self.agent.selectAction(state)
            if q != -100000:
                qSum += q
                qActions += 1

            obs, reward, done, info = self.env.step(action)
            self.env.render()

            if info['x_pos'] < lastX:
                reward -= 1
            if info['flag_get']:
                reward += 10

            if info['x_pos'] > maxDistance:
                maxDistance = info['x_pos']

            nextState = np.array(obs).transpose(3, 1, 2, 0)
            rewardsSum = np.add(rewardsSum, reward)

            self.agent.addMemory((state, action, reward, nextState, done))
            loss = self.agent.trainDQN()
            state = nextState
            lossSum += loss

        if self.current_episode % 200 == 0:
            self.agent.save(self.current_episode)

        print("now epsilon is {}, the reward is {} with loss {} in episode {}".format(
            self.agent.epsilon, rewardsSum, lossSum, self.current_episode))

        self.episode_score.append(rewardsSum)
        self.episode_qs.append(qSum/qActions)
        self.episode_distance.append(maxDistance)
        self.episode_loss.append(lossSum)

    def plot(self):
        self.ax[0][0].title.set_text('Training Score')
        self.ax[0][0].set_xlabel('Episode')
        self.ax[0][0].set_ylabel('Score')
        self.ax[0][0].plot(self.episode_score, 'b')

        self.ax[0][1].title.set_text('Training Distance')
        self.ax[0][1].set_xlabel('Episode')
        self.ax[0][1].set_ylabel('Distance')
        self.ax[0][1].plot(self.episode_distance, 'g')

        self.ax[1][0].title.set_text('Training Loss')
        self.ax[1][0].set_xlabel('Episode')
        self.ax[1][0].set_ylabel('Loss')
        self.ax[1][0].plot(self.episode_loss, 'r')

        self.ax[1][1].title.set_text('Training Q Vals')
        self.ax[1][1].set_xlabel('Episode')
        self.ax[1][1].set_ylabel('Qs')
        self.ax[1][1].plot(self.episode_qs, 'c')
        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(.001)
