import gym
import cv2
import os
import numpy as np
from gym import spaces

os.environ.setdefault('PATH', '')
cv2.ocl.setUseOpenCL(False)

"""
Mainly derived from OpenAI Baselines
https://github.com/openai/baselines/blob/master/baselines/common/wrappers.py
"""


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            # we only remember the latest two obs, and take the maximum pixel values later on.
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=64, height=64, grayscale=True):
        """ Warp frames to 84x84 as done in the Nature paper and later work.
        In the paper, the image size is 64 x 64 x 3
        """
        gym.ObservationWrapper.__init__(self, env)
        self._w = width
        self._h = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._h, self._w, 1), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._h, self._w, 3), dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._w, self._h), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


def wrap_deepmind(env, num_repeat_action=1, grayscale=False):
    """ Configure environment for raw image observation in the simulation """
    env = WarpFrame(env, grayscale=grayscale)
    if num_repeat_action > 1:
        env = MaxAndSkipEnv(env=env, skip=num_repeat_action)
    return env
