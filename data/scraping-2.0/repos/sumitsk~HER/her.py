# modified from OPENAI baselines version

import numpy as np


class HER_sampler(object):
    def __init__(self, replay_k, reward_fun):
        # replay strategy is future
        self.future_p = 1 - (1. / (1 + replay_k))
        self.reward_fun = reward_fun

    def sample(self, episode_batch, batch_size_in_transitions):
        # episode_batch is {key: array(buffer_size x T x dim_key)}

        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        # if episode_idxs is None or t_samples is None:
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. 
        probs = np.random.uniform(size=batch_size)
        her_indexes = np.where(probs < self.future_p)[0]

        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected HER transitions (as defined by her_indexes).
        # For the other transitions, keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = self.reward_fun(**reward_params)
        # her transition rewards are not always 0 
        
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions
