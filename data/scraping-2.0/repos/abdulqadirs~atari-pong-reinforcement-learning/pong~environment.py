#taken from openai baseline wrappers
#https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

import gym
import numpy as np
import torch
import torchvision.transforms as transforms

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)
    

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=224, height=224, grayscale=False):
        """
        Warp frames to 84x84 and converts to pytorh tensor
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._num_colors = None
        if self._grayscale:
            self._num_colors = 1
        else:
            self._num_colors = 3

    def observation(self, obs):
        #pytorch order is (C, H, W)
        obs = obs.transpose(2, 0, 1)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((self._height,self._width)),
                                        transforms.Grayscale(num_output_channels=self._num_colors),
                                        transforms.ToTensor()])

        obs = transform(obs)
        #add dimension for batch (batch_size, channels, height, width)
        obs = obs.unsqueeze(0)

        return obs



def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)

    return env
