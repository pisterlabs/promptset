"""Replay buffer adapted from OpenAI Baselines"""

import numpy as np
import pdb

class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)

class ReplayBuffer(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, _, training=True):
        if not training:
            return

        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)

    @property
    def nb_entries(self):
        return len(self.observations0)

class HERBuffer(ReplayBuffer):
    def __init__(self, limit, action_shape, observation_shape, obs_to_goal, goal_slice, reward_fn):
        """Replay buffer that does Hindsight Experience Replay
        obs_to_goal is a function that converts observations to goals
        goal_slice is a slice of indices of goal in observation
        """
        ReplayBuffer.__init__(self, limit, action_shape, observation_shape)

        self.obs_to_goal = obs_to_goal
        self.goal_slice = goal_slice
        self.reward_fn = reward_fn
        self.data = [] # stores current episode

    def flush(self):
        """Dump the current data into the replay buffer with (final) HER"""
        if not self.data:
            return

        for obs0, action, reward, obs1 in self.data:
            obs0, action, reward, obs1 = obs0.copy(), action.copy(), reward.copy(), obs1.copy()
            super().append(obs0, action, reward, obs1, None)
        final_obs = self.data[-1][-1]
        her_goal = self.obs_to_goal(final_obs)
        for obs0, action, reward, obs1 in self.data:
            obs0, action, reward, obs1 = obs0.copy(), action.copy(), reward.copy(), obs1.copy()
            obs0[self.goal_slice] = her_goal
            obs1[self.goal_slice] = her_goal
            reward =self.reward_fn(obs1)
            super().append(obs0, action, reward, obs1, None)
        self.data = []

    def append(self, obs0, action, reward, obs1, _, training=True):
        if not training:
            return

        self.data.append((obs0, action, reward, obs1))

    @property
    def nb_entries(self):
        return len(self.observations0)
