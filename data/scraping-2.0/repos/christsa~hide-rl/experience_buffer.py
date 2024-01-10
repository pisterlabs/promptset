import numpy as np
from segment_tree import SumSegmentTree, MinSegmentTree

class ExperienceBuffer():
    def __init__(self, max_buffer_size, batch_size, FLAGS, env, layer_number, device, action_labels=False):
        import torch
        self.size = 0
        self.counter = 0
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.device = device
        self.action_labels = action_labels

        self.state_dim = env.state_dim
        if layer_number == 0:
            self.action_dim = env.action_dim
        else:
            self.action_dim = env.subgoal_dim
        if layer_number == FLAGS.layers-1 or (layer_number == FLAGS.layers-2 and FLAGS.oracle):
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim
        
        self.fields = {
            'states': torch.zeros(max_buffer_size, self.state_dim, dtype=torch.float32, device=self.device),
            'actions': torch.zeros(max_buffer_size, self.action_dim, dtype=torch.float32, device=self.device),
            'rewards': torch.zeros(max_buffer_size, dtype=torch.float32, device=self.device),
            'next_states': torch.zeros(max_buffer_size, self.state_dim, dtype=torch.float32, device=self.device),
            'goals': torch.zeros(max_buffer_size, self.goal_dim, dtype=torch.float32, device=self.device),
            'terminals': torch.zeros(max_buffer_size, dtype=torch.float32, device=self.device),
        }
        if FLAGS.sl_oracle and layer_number == FLAGS.layers-1:
            self.fields['action_labels'] = torch.zeros(max_buffer_size, self.action_dim, dtype=torch.float32, device=self.device)
        self.vpn = FLAGS.vpn and layer_number == FLAGS.layers-1

    def add(self, experience):
        assert len(experience) == 9, 'Experience must be of form (s, a, r, s, g, t, hindsight, action_labels, images\')'
        assert type(experience[5]) == bool

        if self.size < self.max_buffer_size:
            self._append_experience(self.counter, experience)
            self.size += 1
            self.counter += 1
        else:
            self._append_experience(self.counter % self.max_buffer_size, experience)
            self.counter = ((self.counter+1) % self.max_buffer_size)

    def _append_experience(self, id, experience):
        import torch
        self.fields['states'][id] = experience[0]
        self.fields['actions'][id] = experience[1]
        self.fields['rewards'][id] = experience[2]
        self.fields['next_states'][id] = experience[3]
        self.fields['goals'][id] = experience[4]
        self.fields['terminals'][id] = experience[5]
        if self.action_labels:
            self.fields['action_labels'][id] = experience[7]
        if self.vpn:
            if 'images' not in self.fields:
                self.fields['images'] = torch.zeros(self.max_buffer_size, experience[8].shape[0], experience[8].shape[1], experience[8].shape[2], dtype=torch.float32, device=self.device)
            self.fields['images'][id] = experience[8]

    def get_batch(self):
        dist = np.random.randint(0, high=self.size, size=min(self.size, self.batch_size))
        data = self.encode_batch(dist)
        return dist, data, None

    def encode_batch(self, idxes):
        import torch
        idxes = torch.tensor(idxes, device=self.device, dtype=torch.int64)
        states = self.fields['states'][idxes]
        actions = self.fields['actions'][idxes]
        rewards = self.fields['rewards'][idxes]
        next_states = self.fields['next_states'][idxes]
        goals = self.fields['goals'][idxes]
        terminals = self.fields['terminals'][idxes]
        if self.action_labels:
            action_labels = self.fields['action_labels'][idxes]
        else:
            action_labels = None
        if self.vpn:
            images = self.fields['images'][idxes]
        else:
            images = None
        return (states, actions, rewards, next_states, goals, terminals, action_labels, images)

    def batch_update(self, *args, **kwargs):
        pass


# From OpenAI Baselines
class PrioritizedReplayBuffer(ExperienceBuffer):
    def __init__(self, size, batch_size, alpha=0.6, **kwargs):
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
        super(PrioritizedReplayBuffer, self).__init__(size, batch_size, **kwargs)
        assert alpha >= 0
        self._alpha = alpha
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self.counter % self.max_buffer_size
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.size - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = np.random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def get_batch(self):
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
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        import torch
        beta = self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        assert beta > 0

        idxes = self._sample_proportional(self.batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.size) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.size) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.expand_dims(np.array(weights), axis=1)
        encoded_sample = self.encode_batch(idxes)
        return idxes, encoded_sample, torch.tensor(weights, device=self.device, dtype=torch.float32)

    def batch_update(self, idxes, priorities):
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
        priorities = priorities.cpu().numpy().astype(np.float64) + 1e-6
        assert (priorities > 0).all()
        self._max_priority = max(self._max_priority, np.max(priorities))
        priorities = priorities ** self._alpha
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < self.size
            self._it_sum[idx] = priority
            self._it_min[idx] = priority
