import torch
from torch import FloatTensor, LongTensor, BoolTensor
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T

import datetime
import random
from collections import namedtuple, deque
from typing import NamedTuple
import numpy as np
from gym.wrappers import FrameStack
from gym.spaces import Box
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import math
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Transition(NamedTuple):
    currStates: FloatTensor
    actions: LongTensor
    rewards: FloatTensor
    nextStates: FloatTensor
    dones: LongTensor

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self._input_shape = input_shape
        self._num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x).view(x.size()[0], -1)
        return self.fc(x)

    @property
    def feature_size(self):
        x = self.features(torch.zeros(1, *self._input_shape))
        return x.view(1, -1).size(1)

class DQNAgent:
    def __init__(self, stateShape, actionSpace, numPicks, memorySize, sync=1000, burnin=100, alpha=0.00025, epsilon=1, epsilon_decay=100000, epsilon_min=0.01, gamma=0.95, checkpoint=None):
        self.numPicks = numPicks
        self.replayMemory = deque(maxlen=memorySize)
        self.stateShape = stateShape
        self.actionSpace = actionSpace

        self.step = 0


        self.sync = sync
        self.burnin = burnin
        self.alpha = alpha
        self.epsilon_start = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        if checkpoint != None:
            self.trainNetwork = torch.load(checkpoint).to(device)
        else:
            self.trainNetwork = DQN(stateShape, actionSpace.n).to(device)

        self.targetNetwork = DQN(stateShape, actionSpace.n).to(device)
        self.targetNetwork.load_state_dict(self.trainNetwork.state_dict())
        self.optimizer = optim.Adam(self.targetNetwork.parameters(), self.alpha)
        self.lossfn = torch.nn.SmoothL1Loss()

    def trainDQN(self):
        if len(self.replayMemory) <= self.numPicks or len(self.replayMemory) <= self.burnin:
            return 0

        #indices = np.random.choice([i for i in range(len(self.replayMemory))], self.numPicks, replace=False)
        samples = random.sample(self.replayMemory, self.numPicks)
        batch = Transition(*zip(*samples))
        currStates, actions, rewards, nextStates, done = batch

        rewards = torch.stack(rewards).squeeze().to(device)
        actions = torch.stack(actions).squeeze().to(device)
        done = torch.stack(done).squeeze().to(device)
        currStates = torch.stack(currStates).to(device)
        nextStates = torch.stack(nextStates).to(device)

        Q_currents = self.trainNetwork(currStates)[np.arange(self.numPicks), actions]
        Q_futures = self.targetNetwork(nextStates).max(1).values

        Q_currents_improved = rewards + (1-done) * Q_futures * self.gamma
        
        loss = self.lossfn(Q_currents, Q_currents_improved)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def selectAction(self, state):
        self.step += 1

        q_value = -100000
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            state = torch.FloatTensor(np.float32(state)).squeeze().unsqueeze(0).to(device)
            preds = self.trainNetwork(state)
            action = torch.argmax(preds, axis=1).item()
            q_value =torch.max(preds, axis=1)
        return action, q_value

    def addMemory(self, state, action, reward, next_state, done):
        self.replayMemory.append((state, action, reward, next_state, done))

    def save(self, ep):
        save_path = (
            f"./mario_torch_{int(ep)}.chkpt"
        )
        torch.save(self.trainNetwork, save_path)
        print(f"MarioNet saved to {save_path} done!")

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

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

        self.env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

        # Apply Frame Wrappers
        self.env = SkipFrame(self.env, 4)
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, 84)
        self.env = FrameStack(self.env, 4)

        self.agent = DQNAgent(stateShape=(4, 84, 84), actionSpace=self.env.action_space, numPicks=32, memorySize=20000, epsilon=epsilon, checkpoint=checkpoint)

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

        state = np.array(self.env.reset())
        maxDistance = -1000000

        while not done:
            action, q = self.agent.selectAction(state)
            '''
            if q != -100000:
                qSum += q
                qActions += 1
            '''
            obs, reward, done, info = self.env.step(action)

            if info['x_pos'] > maxDistance:
                maxDistance = info['x_pos']

            next_state = np.array(obs)
            rewardsSum = np.add(rewardsSum, reward)

            self.agent.addMemory(FloatTensor(state), LongTensor([action]), FloatTensor([reward]), FloatTensor(next_state), LongTensor([done]))
            loss = self.agent.trainDQN()
            state = next_state
            lossSum += loss
            
            if self.agent.step % self.agent.sync == 0:
                self.agent.targetNetwork.load_state_dict(self.agent.trainNetwork.state_dict())
            
            self.agent.epsilon = self.agent.epsilon_min + (self.agent.epsilon_start - self.agent.epsilon_min) * math.exp(-1 * ((self.agent.step + 1) / self.agent.epsilon_decay))
                
        if self.current_episode % 200 == 0:
            self.agent.save(self.current_episode)
                
        print("now epsilon is {}, the reward is {} with loss {} in episode {}, step {}, dist {}".format(
            self.agent.epsilon, rewardsSum, lossSum, self.current_episode, self.agent.step, maxDistance))

        self.episode_score.append(rewardsSum)
        self.episode_qs.append(qSum/qActions)
        self.episode_distance.append(maxDistance)
        self.episode_loss.append(lossSum)

agent = MarioBaseline(10000, None, 0, 1)
agent.train()