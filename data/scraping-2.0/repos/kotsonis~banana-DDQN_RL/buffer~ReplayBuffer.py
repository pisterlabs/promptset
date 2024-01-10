# ReplayBuffer and PrioritizedReplayBuffer classes
# from OpenAI : https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import numpy as np
import torch
import random
from buffer.segmenttrees import SumSegmentTree, MinSegmentTree
from collections import namedtuple, deque
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    """ Experience Replay Buffer class """
    def __init__(self, size:int):
        """Create simple Replay circular buffer as a list"""
        self._buffer = []
        self._maxsize = size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self._next_idx = 0      # next available index for circular buffer is at start

    def __len__(self):
        return len(self._buffer)

    def add(self, state, action, reward, next_state, done):
        """ Add a new experience to replay buffer """
        data = self.experience(state, action, reward, next_state, done)
        if self._next_idx >= len(self._buffer):         
            # when we are still filling the buffer to capacity
            self._buffer.append(data)
        else:
            # overwrite data at index
            self._buffer[self._next_idx] = data         
        #increment buffer index and loop to beginning when needed
        self._next_idx = int((self._next_idx + 1) % self._maxsize)
        
    def _encode_sample(self, idxes):
        "encode batch of experiences indexed by idxes from buffer"
        states, actions, rewards, next_states, dones = [], [], [], [],[]
        for idx in idxes:
            states.append(self._buffer[idx].state)
            actions.append(self._buffer[idx].action)
            rewards.append(self._buffer[idx].reward)
            next_states.append(self._buffer[idx].next_state)
            dones.append(self._buffer[idx].done)
        states = torch.tensor(states).float().to(device)
        actions = torch.tensor(actions).long().unsqueeze(1).to(device)
        rewards = torch.tensor(rewards).float().unsqueeze(1).to(device)
        next_states = torch.tensor(next_states).float().to(device)
        dones = torch.tensor(dones).float().unsqueeze(1).to(device)
        return (states, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        """Sample a random batch of experiences."""
        idxes = [random.randint(0, len(self._buffer) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class PrioritizedReplayBuffer(ReplayBuffer):
    """ A Prioritized according to TD Error replay buffer """
    def __init__(self, size: int, batch_size: int, alpha: float):
        """Create Prioritized(alpha=0 -> no priority) Replay circular buffer as a list"""
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0, "negative alpha not allowed"
        self._alpha = alpha
        self._batch_size = batch_size

        # find minimum power of 2 size for segment trees
        st_capacity = 1
        while st_capacity < size:
            st_capacity *= 2

        self._st_sum = SumSegmentTree(st_capacity)
        self._st_min = MinSegmentTree(st_capacity)
        # set priority with which new experiences will be added. 1.0 means they have highest chance of being sampled
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx                # obtain next available index to store at from the replay buffer parent class

        super().add(*args, **kwargs)        # add to the replay buffer
        self._st_sum[idx] = self._max_priority ** self._alpha   # put it in the sum tree with max priority
        self._st_min[idx] = self._max_priority ** self._alpha   # put it in the min tree with max priority

    def _sample_proportional(self, batch_size: int):
        """ sample uniformly within `batch_size` segments """
        results = []
        p_total = self._st_sum.sum(0, len(self._buffer) - 1)       # get total sum of priorites in the whole replay buffer
        every_range_len = p_total / batch_size                      # split the total sum of priorities into batch_size segments
        for i in range(batch_size):
            # generate a random cummulative sum of priorites within this segment
            mass = random.random() * every_range_len + i * every_range_len 
            #Find index in the array of sampling probabilities such that sum of previous values is mass
            idx = self._st_sum.find_prefixsum_idx(mass)             
            results.append(idx)
        return results

    def sample(self, batch_size:int, beta:float):
        """ sample a batch of experiences from memory and also returns importance weights and idxes of sampled experiences"""
        assert beta > 0
        idxes = self._sample_proportional(batch_size)
        weights = []
        # find maximum weight factor, ie. smallest P(i) since we are dividing by this
        p_sum = self._st_sum.sum()
        p_min = self._st_min.min() / p_sum
        max_weight = (p_min * len(self._buffer)) ** (-beta)
        
        for idx in idxes:
            # Compute importance-sampling weight (w_i) and append to weights
            #              priority of transition
            # P(i) = -------------------------------------
            #        sum of priorities for all transitions
            #       |    1      |^beta
            # w_i = | --------- |
            #       |  N * P(i) |
            # and then normalize by the maximum weight
            # w_j =  w_i/max_weight
            p_sample = self._st_sum[idx] / p_sum
            weight_sample = (p_sample * len(self._buffer)) ** (-beta) 
            weights.append(weight_sample / max_weight)
        #expand weights dimension from (batch_size,) to (batch_size,1)
        weights_t = torch.tensor(weights).unsqueeze(1).to(device)
        
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights_t, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to transitions at the sampled idxes denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._buffer)
            self._st_sum[idx] = priority ** self._alpha     # update value and parent values in sum-tree
            self._st_min[idx] = priority ** self._alpha     # update value and parent values in min-tree

            self._max_priority = max(self._max_priority, priority)