# Replay memory with prioritized replay using segmentation trees
# Modified on top of the code from openai baselines:
#     https://github.com/openai/baselines/tree/master/baselines
# Indexing within segmentation trees: starting from 1 (i.e. 0th element unused)

import numpy as np
import random
import operator
from settings import *

# Segmentation Tree
class SegmentTree(object):
    def __init__(self, segTree_capacity, operation, neutral_element):
        # neutral_element: float('-inf') for max and 0 for sum.
        assert segTree_capacity > 0 and segTree_capacity & (segTree_capacity - 1) == 0, "segTree_capacity must be positive and a power of 2."
        self._segTree_capacity = segTree_capacity
        self._value = [neutral_element for _ in range(2 * segTree_capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._segTree_capacity
        if end < 0:
            end += self._segTree_capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._segTree_capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._segTree_capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx], self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._segTree_capacity
        return self._value[self._segTree_capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, segTree_capacity):
        super(SumSegmentTree, self).__init__(segTree_capacity=segTree_capacity, operation=operator.add, neutral_element=0.0)

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """ Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._segTree_capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._segTree_capacity

class MinSegmentTree(SegmentTree):
    def __init__(self, segTree_capacity):
        super(MinSegmentTree, self).__init__(segTree_capacity=segTree_capacity, operation=min, neutral_element=float('inf'))

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""
        return super(MinSegmentTree, self).reduce(start, end)

"""Base class for Atari replay memory"""
class Replay_Memory_Atari():
    def __init__(self, size):
        self.transactions_storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self.transactions_storage)

    def add(self, state, action, reward, episode_done):
        # only save the last frame of the state, in memory efficient format: uint8
        state_last_frame = (state[-1]*255).astype(np.uint8)
        data = (state_last_frame, action, reward, episode_done, self.episode_t)
        if episode_done:
            self.episode_t = 1
        else:
            self.episode_t += 1
        if self._next_idx >= len(self.transactions_storage):
            self.transactions_storage.append(data)
        else:
            self.transactions_storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    # To reconstruct state with HISTORY_LENGTH stacked frames
    # also called within evaluate.py
    def get_current_and_next_state(self, id):
        state_last_frame, _, _, _, episode_t = self.transactions_storage[id]
        frames = []
        # Fill the first HISTORY_LENGTH number of frames for the current state
        if episode_t >= HISTORY_LENGTH:
            for i in range(1, HISTORY_LENGTH+1):
                frames.append(self.transactions_storage[id+i-HISTORY_LENGTH][0])
        else:
            blank_frame = np.zeros_like(state_last_frame)
            for i in range(HISTORY_LENGTH - episode_t):
                frames.append(blank_frame)
            for i in range(1, episode_t+1):
                frames.append(self.transactions_storage[id+i-episode_t][0])
        # Fill the last frame for the next state
        # If at the time its already terminated, doesn't matter as the frame then wouldn't be used
        frames.append(self.transactions_storage[(id + 1)%len(self.transactions_storage)][0])
        current_state = np.array(frames[:HISTORY_LENGTH])/255
        next_state = np.array(frames[1:HISTORY_LENGTH+1])/255
        assert np.shape(current_state) == np.shape(next_state) == (HISTORY_LENGTH, 84, 84)
        return current_state, next_state

    def _encode_sample(self, idxes):
        states, actions, rewards, next_states, episode_dones = [], [], [], [], []
        for i in idxes:
            data = self.transactions_storage[i]
            _, action, reward, episode_done, _ = data
            state, next_state = self.get_current_and_next_state(i)
            states.append(state)
            actions.append(action)
            rewards.append(max(min(reward,1),-1))
            next_states.append(next_state)
            episode_dones.append(episode_done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(episode_dones)

class Prioritized_Replay_Memory_Atari(Replay_Memory_Atari):
    def __init__(self, size):
        super(Prioritized_Replay_Memory_Atari, self).__init__(size)
        self._alpha = 0.6
        segTree_capacity = 1
        while segTree_capacity < size: # ensure the segTree_capacity is a power of 2
            segTree_capacity *= 2
        self._sum_segTree = SumSegmentTree(segTree_capacity)
        self._min_segTree = MinSegmentTree(segTree_capacity)
        self._max_priority = 1.0
        self._priority_delta = 1e-6
        self.batch_size = 32
        self.episode_t = 1

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._sum_segTree[idx] = self._max_priority ** self._alpha
        self._min_segTree[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self):
        res = []
        p_total = self._sum_segTree.sum(0, len(self.transactions_storage) - 1)
        every_range_len = p_total / self.batch_size
        for i in range(self.batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._sum_segTree.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, beta):
        """ Returns:
        (states_batch, actions_batch, rewwards_batch, next_states_batch, whether_dones_batch,
        importance_weights_batch, idxes_batch) """
        assert beta > 0
        idxes = self._sample_proportional()

        weights = []
        p_min = self._min_segTree.min() / self._sum_segTree.sum()
        max_weight = (p_min * len(self.transactions_storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._sum_segTree[idx] / self._sum_segTree.sum()
            weight = (p_sample * len(self.transactions_storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights) 
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        priorities += self._priority_delta # for non-zero probability of sampling transactions with 0 TD error
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.transactions_storage)
            self._sum_segTree[idx] = priority ** self._alpha
            self._min_segTree[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

class Uniform_Replay_Memory_Atari(Replay_Memory_Atari):
    def __init__(self, size):
        super(Uniform_Replay_Memory_Atari, self).__init__(size)
        self.batch_size = 32
        self.episode_t = 0  # Not used in this child-class

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)

    def _sample_uniform(self):
        res = np.random.choice(len(self.transactions_storage), size=self.batch_size, replace=False)
        return res

    def sample(self):
        """ Returns:
        (states_batch, actions_batch, rewwards_batch, next_states_batch, whether_dones_batch) """
        idxes = self._sample_uniform()
        encoded_sample = self._encode_sample(idxes)
        return list(encoded_sample)

"""Base class for gym replay memory"""
class Replay_Memory_Gym():
    def __init__(self, size):
        self.transactions_storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self.transactions_storage)

    def add(self, state, action, reward, episode_done):
        data = (state, action, reward, episode_done)
        if self._next_idx >= len(self.transactions_storage):
            self.transactions_storage.append(data)
        else:
            self.transactions_storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    # Also called from evalaute.py for evaluating Q over states
    # Don't really need this function for the simpler gym memory, follow the same function for atari memory
    def get_current_and_next_state(self, id):
        state, _, _, _ = self.transactions_storage[id]
        # even if the current state terminates (and thus next state is irrelevant), it wouldn't matter in Q updates
        next_state, _, _, _ = self.transactions_storage[(id+1)%len(self.transactions_storage)]
        return state, next_state

    def _encode_sample(self, idxes):
        states, actions, rewards, next_states, episode_dones = [], [], [], [], []
        for i in idxes:
            _, action, reward, episode_done = self.transactions_storage[i]
            state, next_state = self.get_current_and_next_state(i)
            states.append(state)
            actions.append(action)
            rewards.append(max(min(reward,1),-1))
            next_states.append(next_state)
            episode_dones.append(episode_done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(episode_dones)

class Prioritized_Replay_Memory_Gym(Replay_Memory_Gym):
    def __init__(self, size):
        super(Prioritized_Replay_Memory_Gym, self).__init__(size)
        self._alpha = 0.6
        segTree_capacity = 1
        while segTree_capacity < size: # ensure the segTree_capacity is a power of 2
            segTree_capacity *= 2
        self._sum_segTree = SumSegmentTree(segTree_capacity)
        self._min_segTree = MinSegmentTree(segTree_capacity)
        self._max_priority = 1.0
        self._priority_delta = 1e-6
        self.batch_size = 32
        self.episode_t = 1

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._sum_segTree[idx] = self._max_priority ** self._alpha
        self._min_segTree[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self):
        res = []
        p_total = self._sum_segTree.sum(0, len(self.transactions_storage) - 1)
        every_range_len = p_total / self.batch_size
        for i in range(self.batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._sum_segTree.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, beta):
        """ Returns:
        (states_batch, actions_batch, rewwards_batch, next_states_batch, whether_dones_batch,
        importance_weights_batch, idxes_batch) """
        assert beta > 0
        idxes = self._sample_proportional()

        weights = []
        p_min = self._min_segTree.min() / self._sum_segTree.sum()
        max_weight = (p_min * len(self.transactions_storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._sum_segTree[idx] / self._sum_segTree.sum()
            weight = (p_sample * len(self.transactions_storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights) 
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        priorities += self._priority_delta # for non-zero probability of sampling transactions with 0 TD error
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.transactions_storage)
            self._sum_segTree[idx] = priority ** self._alpha
            self._min_segTree[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)


class Uniform_Replay_Memory_Gym(Replay_Memory_Gym):
    def __init__(self, size):
        super(Uniform_Replay_Memory_Gym, self).__init__(size)
        self.batch_size = 32
        self.episode_t = 0  # Not used in this child-class

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)

    def _sample_uniform(self):
        res = np.random.choice(len(self.transactions_storage), size=self.batch_size, replace=False)
        return res

    def sample(self):
        """ Returns:
        (states_batch, actions_batch, rewwards_batch, next_states_batch, whether_dones_batch) """
        idxes = self._sample_uniform()
        encoded_sample = self._encode_sample(idxes)
        return list(encoded_sample)
