"""
This implementation is inspired from OpenAI
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

from collections import deque
from typing import Type

import gym
import numpy as np
from gym import spaces, Env


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = self._make_stacked()
            self._frames = None
        return self._out

    def _make_stacked(self):
        return np.concatenate(self._frames, axis=-1)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class LazyVStackedFrames(LazyFrames):
    def _make_stacked(self):
        return np.vstack(self._frames)


class FrameStack(gym.Wrapper):
    def __init__(self, env: Env, k: int, lazy_class: Type = LazyFrames):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        """
        gym.Wrapper.__init__(self, env)
        self.lazy_class = lazy_class
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype
        )

    def step(self, action: int):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return self.lazy_class(list(self.frames))
