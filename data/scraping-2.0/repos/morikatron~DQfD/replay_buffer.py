# This code is based on code from OpenAI baselines. (https://github.com/openai/baselines)
import numpy as np
import random

from common.segment_tree_sb import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.demo_len = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, is_demo, obs_tpn=None, reward_n=None, done_n=None):
        data = (obs_t, action, reward, obs_tp1, done, is_demo, obs_tpn, reward_n, done_n)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        elif self._storage[self._next_idx][5]:
            self._next_idx = self.demo_len
            self._storage[self._next_idx] = data
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        data = self._storage[0]
        ob_dtype = data[0].dtype
        ac_dtype = data[1].dtype
        obses_t, actions, rewards, obses_tp1, dones, is_demos, obses_tpn, rewards_n, dones_n = [], [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, is_demo, obs_tpn, reward_n, done_n = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            is_demos.append(is_demo)
            # n_step
            obses_tpn.append(np.array(obs_tpn, copy=False))
            rewards_n.append(reward_n)
            dones_n.append(done_n)
        if data[6] is None:
            return np.array(obses_t, dtype=ob_dtype), np.array(actions, dtype=ac_dtype), np.array(rewards, dtype=np.float32), \
                   np.array(obses_tp1, dtype=ob_dtype), np.array(dones, dtype=np.float32), np.array(is_demos, dtype=np.float32), \
                   None, None, None
        else:
            return np.array(obses_t, dtype=ob_dtype), np.array(actions, dtype=ac_dtype), np.array(rewards, dtype=np.float32), \
                   np.array(obses_tp1, dtype=ob_dtype), np.array(dones, dtype=np.float32), np.array(is_demos, dtype=np.float32), \
                   np.array(obses_tpn, dtype=ob_dtype), np.array(rewards_n, dtype=np.float32), np.array(dones_n, dtype=np.float32)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    # def _sample_proportional(self, batch_size):
    #     res = []
    #     p_total = self._it_sum.sum(0, len(self._storage) - 1)
    #     every_range_len = p_total / batch_size
    #     for i in range(batch_size):
    #         mass = random.random() * every_range_len + i * every_range_len
    #         idx = self._it_sum.find_prefixsum_idx(mass)
    #         res.append(idx)
    #     return res

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    # def sample(self, batch_size, beta):
    #     assert beta > 0
    #
    #     idxes = self._sample_proportional(batch_size)
    #
    #     weights = []
    #     p_min = self._it_min.min() / self._it_sum.sum()
    #     max_weight = (p_min * len(self._storage)) ** (-beta)
    #
    #     for idx in idxes:
    #         p_sample = self._it_sum[idx] / self._it_sum.sum()
    #         weight = (p_sample * len(self._storage)) ** (-beta)
    #         weights.append(weight / max_weight)
    #     weights = np.array(weights, dtype=np.float32)
    #     encoded_sample = self._encode_sample(idxes)
    #     return tuple(list(encoded_sample) + [weights, idxes])

    def sample(self, batch_size, beta = 0):
        """
        Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        weights = np.array(weights, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    # def update_priorities(self, idxes, priorities):
    #     assert len(idxes) == len(priorities)
    #     for idx, priority in zip(idxes, priorities):
    #         assert priority > 0
    #         assert 0 <= idx < len(self._storage)
    #         self._it_sum[idx] = priority ** self._alpha
    #         self._it_min[idx] = priority ** self._alpha
    #
    #         self._max_priority = max(self._max_priority, priority)

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self._storage)
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))