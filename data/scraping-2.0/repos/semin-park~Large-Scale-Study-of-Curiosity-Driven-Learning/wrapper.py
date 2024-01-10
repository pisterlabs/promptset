from collections import deque

import gym
from baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv
import numpy as np
import cv2

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

# Adapted from OpenAI's baselines.common.atari_wrappers.py
class ChannelFirstFrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Identical to OpenAI's FrameStack, except that this is channel first.
        """
        super(ChannelFirstFrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((shp[-1] * k,) + shp[:-1]), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        ob = ob.transpose(2,0,1)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = ob.transpose(2,0,1)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames(object):
    def __init__(self, frames):
        """Concatenates channel first"""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[0]

    def frame(self, i):
        return self._force()[i]

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(ScaledFloatFrame, self).__init__(env)

    def observation(self, obs):
        return obs / 255.0

def make_env(env_name='PongNoFrameskip-v4', size=42, skip=4, is_train=True):
    env = gym.make(env_name)
    env = NoopResetEnv(env, noop_max=300)
    if is_train:
        env = MaxAndSkipEnv(env, skip=skip)
    env = WarpFrame(env, width=size, height=size, grayscale=True) # obs_space is now (84,84,1)
    env = ScaledFloatFrame(env)
    env = ChannelFirstFrameStack(env, 4)
    return env