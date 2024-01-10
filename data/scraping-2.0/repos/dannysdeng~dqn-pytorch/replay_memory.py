from collections import namedtuple
import random
import numpy as np
from utils import SumTree
from utils import SumSegmentTree, MinSegmentTree
from multiprocessing import Pool, Manager

import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity, low_footprint=False, num_workers=1):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.low_footprint = low_footprint
        self.index_list = list(range(capacity))
        self.num_workers = num_workers
        if self.low_footprint and self.num_workers > 1:
            raise NotImplementedError('Multi processing for replay not implemented')

    def _get_transition(self, index):
        """
        For low-footprint. Need to check whether a transition is valid.
            e.g., if next_state is None, it means that it is an end.
            So it could be: 
                [Zero, Zero, s_{t-1}, s_{t}]
            or:
                [Zero, s_{t-2}, s_{t-1}, s_{t}]

        Case 1: 
            A memory item is not valid because it is overridden.
            [s1, s2, s3, s4, ..., sN-1, sN]     --> s1 is not valid  
            [sN+1, s2, s3, s4, ..., sN-1, sN]   --> s2 to s4 is not valid, cuz s1 is removed
            [sN+1, sN+2, s3, s4, ..., sN-1, sN] --> s3 to s6 is not valid, cuz s2 is removed

        Case 2:
            A memory item needs to be concated zeros, because it is an early met terminal state
            [s1, s2_T, s3, s4, ..., sN-1, sN] --> s2_T is valid, but needs to be cat [Zero, Zero, s1, s2_T]

            [sN+1, s2_T, s3, s4, ..., sN-1, sN] --> s2_T is not valid, cuz s1 is removed

            [sN+1, sN+2, s3, s4_T, ..., sN-1, sN] --> s4_T should be valid. As we can do [Zero, Zero, s3, s4_T]

        """
        tran = self.memory[index]
        s0, a, s1, r = tran.state, tran.action, tran.next_state, tran.reward
        #
        state_list      = [s0]
        next_state_list = [s1] # s1 will be None if it is a terminal state
        #
        for i in range(1, 3+1):
            prev_tran = self.memory[index-i]
            s0_prev, s1_prev = prev_tran.state, prev_tran.next_state
            if s1_prev is None:
                break
            state_list.append(s0_prev)
            next_state_list.append(s1_prev)
        #
        state_list      = state_list[::-1]
        next_state_list = next_state_list[::-1]
        # -----------------------------------------
        if s1 is not None and len(next_state_list) < 4:
            next_state_list = [state_list[0]] + next_state_list
            if len(next_state_list) < 4:
                next_state_list = [torch.zeros_like(s0)] * (4-len(next_state_list)) + next_state_list

        if len(state_list) < 4:
            state_list = [torch.zeros_like(s0)] * (4 - len(state_list)) + state_list

        # -----------------------------------------
        state      = torch.cat(state_list, dim=1)      # from [1 x 1 x 84 x 84] to [1 x 4 x 84 x 84]
        next_state = torch.cat(next_state_list, dim=1) if s1 is not None else None # from [1 x 1 x 84 x 84] to [1 x 4 x 84 x 84]
        return Transition(state, a, next_state, r)     

    # args is like def push(self, state, action, next_state, reward), 
    # So Transition(state, action, next_state, reward) becomes what we need to store
    def push(self, *args):
        # ------------------------------------------------------
        if len(self.memory) < self.capacity:
            self.memory.append(None)        
        # ------------------------------------------------------
        if self.low_footprint:
            # Only store the latest frame
            state, action, next_state, reward = args
            if next_state is not None:
                self.memory[self.position] = Transition(state[:, -1:, :, :].cpu(), 
                                                        action.cpu(), 
                                                        next_state[:, -1:, :, :].cpu(),
                                                        reward.cpu())
            else:
                self.memory[self.position] = Transition(state[:, -1:, :, :].cpu(), 
                                                        action.cpu(), 
                                                        None,
                                                        reward.cpu())                
        else:
            self.memory[self.position] = Transition(*args)
        # ------------------------------------------------------
        self.position = (self.position + 1) % self.capacity
        # ------------------------------------------------------
    def sample(self, batch_size):
        if self.low_footprint:
            if len(self.memory) < self.capacity:
                valid_range = self.index_list[3:len(self.memory)]
            else:
                if self.position <= 3:
                    valid_range = self.index_list[self.position+3:]
                else:
                    valid_range = self.index_list[3:self.position] + self.index_list[self.position+3:]
            output_batch = []
            while len(output_batch) < batch_size:
                out_index = random.sample(valid_range, batch_size)
                for index in out_index:
                    this_one = self._get_transition(index)
                    if this_one is not None:
                        output_batch.append(this_one)
                    if len(output_batch) == batch_size:
                        break
            return output_batch
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayBuffer():
    """
    PrioritizedReplayBuffer From OpenAI Baseline
    """
    def __init__(self, size, T_max, learn_start):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        #
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self._sumTree = SumSegmentTree(it_capacity)
        self._minTree = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        #
        self.e     = 0.01
        self.alpha = 0.5 # tradeoff between taking only experience with high-priority samples
        self.beta  = 0.4 # Importance Sampling, from 0.4 -> 1.0 over the course of training
        self.beta_increment = (1 - self.beta) / (T_max - learn_start)
        self.abs_error_clipUpper = 1.0
        self.NORMALIZE_BY_BATCH = False # In openAI baseline, normalize by whole        

    def __len__(self):
        return len(self._storage)

    def push(self, state, action, next_state, reward):
        idx = self._next_idx
        #
        # Setting maximum priority for new transitions. Total priority will be updated
        if next_state is not None:
            data = Transition(state.cpu(), action.cpu(), next_state.cpu(), reward.cpu())
        else:
            data = Transition(state.cpu(), action.cpu(), None, reward.cpu())
        #
        if self._next_idx >= len(self._storage):
            self._storage += data,
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        #
        self._sumTree[idx] = self._max_priority ** self.alpha
        self._minTree[idx] = self._max_priority ** self.alpha

    def sample(self, batch_size):
        # indices = self._sample_proportional(batch_size)
        indices      = []
        batch_sample = []
        weights = []
        # Increase the beta each time we sample a new mini-batch until it reaches 1.0
        self.beta = min(self.beta + self.beta_increment, 1.0)        
        #
        total_priority   = self._sumTree.sum(0, len(self._storage) - 1)
        priority_segment = total_priority / batch_size
        #
        min_priority            = self._minTree.min() / self._sumTree.sum()
        max_weight_ALL_memory   = (min_priority * len(self._storage)) ** (-self.beta)
        #        
        for i in range(batch_size):
            mass = (i + random.random()) * priority_segment
            index  = self._sumTree.find_prefixsum_idx(mass)
            # P(j) --> stochastic priority
            stochastic_p = self._sumTree[index] / total_priority
            this_weight_IS = (stochastic_p * len(self._storage)) ** (-self.beta)
            """
                Importance Sampling Weight:
                     [   1      1        ]^(beta)
                     |  --- * -----------|
                     [   N     prob_min  ]                
            """
            this_weight_IS /= max_weight_ALL_memory
            # Append to list
            weights      += this_weight_IS,
            batch_sample += self._storage[index],
            indices      += index,
            #
        return batch_sample, indices, weights
    def update_priority_on_tree(self, tree_idx, abs_TD_errors):
        assert(len(tree_idx) == len(abs_TD_errors))
        abs_TD_errors  = np.nan_to_num(abs_TD_errors) + self.e
        abs_TD_errors  = abs_TD_errors.tolist()
        #
        for index, priority in zip(tree_idx, abs_TD_errors):
            assert(priority > 0)
            assert(0<=index<=len(self._storage))
            self._sumTree[index] = priority ** self.alpha
            self._minTree[index] = priority ** self.alpha
            #
            self._max_priority = max(self._max_priority, priority)
    #


class PrioritizedReplayBuffer_slow():
    # Deprecated 
    def __init__(self, capacity, T_max, learn_start):
        self.capacity = capacity
        self.count    = 0
        # We may want better data structure: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
        self.memory   = SumTree(capacity) 

        self.e     = 0.01
        self.alpha = 0.5 # tradeoff between taking only experience with high-priority samples
        self.beta  = 0.4 # Importance Sampling, from 0.4 -> 1.0 over the course of training
        self.beta_increment = (1 - self.beta) / (T_max - learn_start)
        self.abs_error_clipUpper = 1.0
        self.NORMALIZE_BY_BATCH = False # In openAI baseline, normalize by whole

    def push(self, state, action, next_state, reward):      
        # Find the max priority. Recall that treeArr is of size 2*capacity - 1. 
        # And all the priorioties lie on the leaves of the tree
        self.count += 1
        self.count = max(self.count, self.capacity)
        all_priority = self.memory.treeArr[-self.memory.capacity:][:self.count]
        max_priority = np.max(all_priority)
        if max_priority == 0:
            max_priority = self.abs_error_clipUpper
        # Setting maximum priority for new transitions. Total priority will be updated
        if next_state is not None:
            transition = Transition(state.cpu(), action.cpu(), next_state.cpu(), reward.cpu())
        else:
            transition = Transition(state.cpu(), action.cpu(), None, reward.cpu())
        self.memory.push(max_priority, transition) 

    def sample(self, batch_size):
        """
        Let N = batch_size

        1. First, divide the range [0, priority_total] into N ranges
        2. Next,  uniformly sample one value per range (out of N ranges)
        3. Then, go and search the SumTree, 
                the transitions with (priority score == sampled values) are retrieved
        4. Finally, calculate Importance Sampling weight, W_is, for each of the element in the minibatch
        """
        n = batch_size
        this_batch = []
        batch_index     = [] # np.empty((n,  ), dtype=np.int32)
        batch_weight_IS = [] # np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment by dividing the ranges
        total_priority   = self.memory.get_total_priority()
        priority_segment = total_priority / batch_size

        # Increase the beta each time we sample a new mini-batch until it reaches 1.0
        self.beta = min(self.beta + self.beta_increment, 1.0)

        # Calculate the max_weight
        all_priority = self.memory.treeArr[-self.memory.capacity:][:self.count]
        prob_min     = min(all_priority) / total_priority
        assert(prob_min > 0)
        """
        N is the batch size

             [     prob_min * N ]^(-beta)             [   1      1        ]^(beta)
             |------------------|             --->    |  --- * -----------|
             [          1       ]                     [   N     prob_min  ]

        """
        # Getting the MAX of importance sampling weight for nomalization
        max_weight_ALL_memory   = (prob_min * n)**(-self.beta)
        max_weight_THIS_BATCH   = -1
        # 
        for i in range(batch_size):
            # A value is sample from each range
            A = A_the_ith_range = priority_segment *  i
            B = B_the_ith_range = priority_segment * (i + 1)
            sampled_value = np.random.uniform(A, B)

            # The transition that corresponds to the "sampled_value" is retrieved
            index, priority, data = self.memory.get_leaf(sampled_value)
            transition = data

            # P(j) --> stochastic priority
            stochastic_p = priority / total_priority

            """
                Importance Sampling Weight:
                     [   1      1        ]^(beta)
                     |  --- * -----------|
                     [   N     prob_min  ]                
            """
            this_weight_IS = (stochastic_p * n) ** (-self.beta)

            if self.NORMALIZE_BY_BATCH and max_weight_THIS_BATCH <= this_weight_IS:
                max_weight_THIS_BATCH = this_weight_IS

            # List append
            batch_weight_IS += this_weight_IS, # batch_weight_IS[i, 0] = this_weight_IS
            batch_index     += index,          #batch_index[i]        = index
            this_batch      += transition,
        #
        batch_weight_IS = np.asarray(batch_weight_IS).T # --> make it 32 x 1
        batch_index     = np.asarray(batch_index)
        # ------------------------------------------------------------------- #
        if self.NORMALIZE_BY_BATCH:
            batch_weight_IS /= max_weight_THIS_BATCH # Kaixin from Berkeley
        else:
            batch_weight_IS /= max_weight_ALL_memory # OpenAI Baseline
        # ------------------------------------------------------------------- #
        return this_batch, batch_index, batch_weight_IS

    def update_priority_on_tree(self, tree_idx, abs_TD_errors):
        """ 
            A bunch of tree indices and a bunch of TD_errors
        """
        abs_TD_errors  = np.nan_to_num(abs_TD_errors)
        abs_TD_errors  += self.e # p_t = |delta_t| + e
        clipped_errors = np.minimum(abs_TD_errors, self.abs_error_clipUpper)
        pt_alpha       = np.power(clipped_errors, self.alpha)
        for index, prob in zip(tree_idx, pt_alpha):
            self.memory.update(index, prob)
    # Remember to deal with EMPTY MEMORY PROBLEM
