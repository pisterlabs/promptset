#part of the code from openai
#https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import numpy as np
import random

import operator
from numba import njit

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient `reduce`
               operation which reduces `operation` over
               a contiguous subsequence of items in the
               array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must for a mathematical group together with the set of
            possible values for array elements.
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = np.array([neutral_element for _ in range(2 * capacity)])
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
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

@njit(parallel=False)
def compiled_setitem_maxtree(idx, val, _value, _capacity):
    idx += _capacity
    _value[idx] = val
    idx //= 2
    while idx >= 1:
        _value[idx] = max(_value[2 * idx], _value[2 * idx + 1]) 
        idx //= 2

class MaxSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MaxSegmentTree, self).__init__(
            capacity=capacity,
            operation=max,
            neutral_element=0. # we assume that all elements are larger than zero
        )
    # the maximum value can be accessed directly by "._value[1]"
    def max(self, start=0, end=None):
        """Returns max(arr[start], ...,  arr[end])"""
        return super(MaxSegmentTree, self).reduce(start, end)
        #return self._value[1]

@njit(parallel=False)
def compiled_setitem_mintree(idx, val, _value, _capacity):
    idx += _capacity
    _value[idx] = val
    idx //= 2
    while idx >= 1:
        _value[idx] = min(_value[2 * idx], _value[2 * idx + 1]) 
        idx //= 2

class MinSegmentTree(SegmentTree):
    def __init__(self, capacity, neutral_element=float("inf")):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=neutral_element
        )

@njit(parallel=False)
def compiled_setitem_sumtree(idx, val, _value, _capacity):
    idx += _capacity
    _value[idx] = val
    idx //= 2
    while idx >= 1:
        _value[idx] = _value[2 * idx] + _value[2 * idx + 1] 
        idx //= 2

@njit(parallel=False)
def compiled_setitem_min_sumtree(idx, min_val, _value, _capacity):
    idx += _capacity
    if min_val > _value[idx]:
        _value[idx] = min_val
        idx //= 2
        while idx >= 1:
            _value[idx] = _value[2 * idx] + _value[2 * idx + 1] 
            idx //= 2

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.
        )
    # the total sum can be accessed directly by "._value[1]"
    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)
        #return self._value[1]

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        return compiled_find_prefixsum_idx(prefixsum, self._capacity, self._value)

@njit(parallel=False)
def compiled_find_prefixsum_idx(prefixsum, _capacity, _value):
    idx = 1
    while idx < _capacity:  # while non-leaf
        if _value[2 * idx] > prefixsum:
            idx = 2 * idx
        else:
            prefixsum -= _value[2 * idx]
            idx = 2 * idx + 1
    return idx - _capacity




class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped. The index of the next transition
            to store can be accessed by "self._next_idx". 
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.cache = None
        self.cached_data = None
        self.indices_replaced_after_caching = []

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = [obs_t, action, reward, obs_tp1, done]

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            assert len(self._storage) == self._maxsize
            self._storage[self._next_idx] = data
            if self.cache is not None:
                self.indices_replaced_after_caching.append(self._next_idx)
        self._next_idx = (self._next_idx + 1) % self._maxsize
        
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            #data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = self._storage[i]
            obses_t.append(obs_t._frames)
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1._frames)
            dones.append(done)
        shp = obs_t._frames[0].shape
        obses_t_obses_tp1 = np.array([obses_t, obses_tp1]).reshape(2, len(idxes), -1, shp[-2], shp[-1]) # their data types are np.uint8
        return obses_t_obses_tp1, np.array(actions, dtype=np.int64), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.float32)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch, next_obs_batch: np.array
            batch of observations, next set of observations seen after executing act_batch
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if self.cache is None:
            # python random.randint is different from np.random.randint; np.random.randint is the same as random.randrange
            idxes = np.random.randint(0, len(self._storage), size = batch_size) 
            #idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
            return self._encode_sample(idxes) + (idxes,)
        else:
            return self.retrieve_cache() 
    
    def _encode_next_state_data(self, idxes):
        obses_tp1 = []
        for i in idxes:
            obs_tp1 = self._storage[i][3]
            obses_tp1.append(obs_tp1._frames)
        obses_tp1 = np.array(obses_tp1)
        return obses_tp1
    
    def sample_next_state_and_cache_indices(self, batch_size): 
        idxes = np.random.randint(0, len(self._storage), size = batch_size) 
        self.cache = (idxes, ) 
        return self._encode_next_state_data(idxes), idxes

    def update_and_store_cached_data(self): 
        assert self.cache is not None
        idxes = self.cache[-1] 
        self.cached_data = self._encode_sample(idxes) + self.cache
        self.indices_replaced_after_caching.clear() 

    def retrieve_cache(self):
        data = self.cached_data 
        self.cache, self.cached_data = None, None
        return data

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, IS_weight_only_smaller, allowed_avg_min_ratio = 10):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self.it_capacity = it_capacity
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_max = MaxSegmentTree(it_capacity)
        self._max_priority = 100. 
        self._max_priority = self._max_priority ** self._alpha
        self.IS_weight_only_smaller = IS_weight_only_smaller 
        if IS_weight_only_smaller:
            self._it_min = MinSegmentTree(it_capacity, neutral_element=self._max_priority)
            self._min_priority = self._max_priority
        assert allowed_avg_min_ratio > 1 or allowed_avg_min_ratio <= 0, "'allowed_avg_min_ratio' ({}) is not within the allowed range.".format(allowed_avg_min_ratio)
        if allowed_avg_min_ratio <= 0: allowed_avg_min_ratio = float("inf")
        self._allowed_avg_min_ratio = float(allowed_avg_min_ratio) # the maximum allowed relative difference between the min and the avg priorities

    def add(self, *args, prio=None, **kwargs): # "prio" stands for priority
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        if prio is None:
            prio = self._max_priority
        else:
            assert prio > 0.
            prio = max(prio ** self._alpha, self._it_sum._value[1]/(len(self._storage)*self._allowed_avg_min_ratio))
        compiled_setitem_sumtree(idx, prio, self._it_sum._value, self.it_capacity)
        super(PrioritizedReplayBuffer, self).add(*args, **kwargs)

    def _sample_proportional(self, batch_size, beta=1.):
        weights, true_weights, idxes = compiled_sample_proportional(batch_size, self._it_sum._value, self._it_sum._capacity, len(self._storage), beta)
        if self.IS_weight_only_smaller:
            # divide the weights by the largest weight possible, which corresponds to the minimal priority 
            weights = weights / ( (self._it_sum._value[1]/len(self._storage)/self._min_priority)**beta )
        else:
            weights = np.minimum(weights, 2.*self._allowed_avg_min_ratio)
        return weights.astype(np.float32), true_weights, idxes

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch, next_obs_batch: np.array
        act_batch: np.array
        rew_batch: np.array
        done_mask: np.array
        weights: np.array
        true_weights: np.array
        idxes: np.array
        """
        assert beta >= 0.
        if self.cache is None:
            weights, true_weights, idxes = self._sample_proportional(batch_size, beta) 
            encoded_sample = self._encode_sample(idxes) 
            return encoded_sample + (weights, true_weights, idxes) 
        else: 
            return self.retrieve_cache() 

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        self._max_priority, clipped_priorities = compiled_update_priorities(idxes, priorities.astype(float),
           self._alpha, self._it_sum._value, self._it_max._value, self.it_capacity, len(self._storage), self._allowed_avg_min_ratio)
        if self.IS_weight_only_smaller:
            self._min_priority = compiled_update_min_priority(idxes, clipped_priorities, self._it_min._value, self.it_capacity)

    def prop_minimum_priorities(self, idxes, minimum_priorities):
        if len(self._storage) < self._maxsize: 
            mask = (idxes>=0)
            if not np.all(mask):
                idxes, minimum_priorities = idxes[mask], minimum_priorities[mask] 
        else: 
            idxes = idxes % self._maxsize
        mask = np.array([self._storage[i][4]!=1. for i in idxes])
        if not np.all(mask):
            idxes, minimum_priorities = idxes[mask], minimum_priorities[mask]
        compiled_update_prop_minimum_priorities(idxes, minimum_priorities, self._alpha, self._it_sum._value, self.it_capacity, len(self._storage), self._allowed_avg_min_ratio)
    
    def sample_next_state_and_cache_indices(self, batch_size, beta): 
        assert beta >= 0.
        self.cache = self._sample_proportional(batch_size, beta)
        idxes = self.cache[-1] 
        return self._encode_next_state_data(idxes), idxes


@njit(parallel=False)
def compiled_update_priorities(idxes, priorities, _alpha, _value, _max_value, _capacity, length, _allowed_avg_min_ratio):
    # change priorities to sampling probabilities first
    priorities = priorities ** _alpha
    for idx, priority in zip(idxes, priorities):
        assert 0 <= idx < length
        assert priority >= 0 # If the priority was smaller than zero, it will become nan at "priorities**_alpha" and fail to pass the assertion here 
        compiled_setitem_maxtree(idx, priority, _max_value, _capacity)
    # this is the maximum of sampling probabilities
    _max_priority = _max_value[1]
    clipped_priorities = np.maximum(priorities, _value[1] / (length * _allowed_avg_min_ratio))
    for idx, clipped_priority in zip(idxes, clipped_priorities):
        compiled_setitem_sumtree(idx, clipped_priority, _value, _capacity)
    return _max_priority, clipped_priorities

@njit(parallel=False)
def compiled_update_prop_minimum_priorities(idxes, minimum_priorities, _alpha, _value, _capacity, length, _allowed_avg_min_ratio):
    # change priorities to sampling probabilities first
    minimum_priorities = minimum_priorities ** _alpha
    clipped_priorities = np.maximum(minimum_priorities, _value[1] / (length * _allowed_avg_min_ratio))
    for idx, clipped_priority in zip(idxes, clipped_priorities):
        assert 0 <= idx < length
        assert clipped_priority >= 0 # If the priority was smaller than zero, it will become nan at "priorities**_alpha" and fail to pass the assertion here 
        compiled_setitem_min_sumtree(idx, clipped_priority, _value, _capacity)
    return

@njit(parallel=False)
def compiled_update_min_priority(idxes, clipped_priorities, _min_value, _capacity):
    for idx, clipped_priority in zip(idxes, clipped_priorities):
        compiled_setitem_mintree(idx, clipped_priority, _min_value, _capacity)
    # this is the minimum of sampling probabilities
    _min_priority = _min_value[1]
    return _min_priority
    
@njit(parallel=False)
def compiled_sample_proportional(batch_size, _value, _capacity, length, beta):
    res = np.zeros(batch_size, dtype = np.int64)
    weights = np.empty(batch_size, dtype = np.float64)
    p_total = _value[1]
    masses = (np.random.random(batch_size) + np.arange(batch_size, dtype=np.float64)) * (p_total / batch_size)
    for i, mass in enumerate(masses):
        idx = compiled_find_prefixsum_idx(mass, _capacity, _value)
        p = _value[idx+_capacity]
        while p == 0.: 
            idx = compiled_find_prefixsum_idx( (random.random()+i) * (p_total / batch_size), _capacity, _value)
            p = _value[idx+_capacity]
        res[i] = idx
        weights[i] = p
    weights = weights*(length/p_total)
    true_weights = 1./weights
    weights = true_weights**beta
    return weights, true_weights.astype(np.float32), res 
