# This file reuses a lot of code from OpenAI baselines/baselines/deepq/replay_buffer.py and
# from a tutorial at https://github.com/jachiam/rl-intro

import numpy as np
import random


class ReplayBuffer:

    def __init__(self, obs_dim, n_acts, size):
        self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, n_acts], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.total_count = 0
        self.size = 0
        self.max_size = size
        print("Initialized ReplayBuffer")

    def store(self, obs, act, rew, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        self.total_count += 1

    def store_batch(self, batch):
        for data in batch:
            self.store(data['obs'], data['act'], data['rew'], data['done'])

    def choose_batch_idxs(self, batch_size, include_most_recent):
        idxs = np.random.choice(self.size, batch_size)
        if include_most_recent:
            idxs[-1] = self.ptr - 1
        return idxs

    def sample(self, batch_size=32, include_most_recent=False):
        idxs = self.choose_batch_idxs(batch_size, include_most_recent)
        return dict(cur_obs=self.obs_buf[idxs],
                    next_obs=self.obs_buf[(idxs + 1) % self.max_size],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])
