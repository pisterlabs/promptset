from typing import List
import collections
import gym
import numpy as np

import atari_utils


class ClipRewardEnv(gym.RewardWrapper):
    """Adapted from OpenAI baselines.

    github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class FrameStack:
    """Implements stacking of `num_frames` last frames of the game.

    Wraps an AtariPreprocessing object.
    """
    def __init__(self,
                 preproc: atari_utils.AtariPreprocessing,
                 num_frames: int):
        self.preproc = preproc
        self.num_frames = num_frames
        self.frames = collections.deque(maxlen=num_frames)

    def reset(self):
        ob = self.preproc.reset()
        for _ in range(self.num_frames):
            self.frames.append(ob)
        return self._get_array()

    def step(self, action: int):
        ob, reward, done, info = self.preproc.step(action)
        self.frames.append(ob)
        return self._get_array(), reward, done, info

    def _get_array(self):
        assert len(self.frames) == self.num_frames
        return np.concatenate(self.frames, axis=-1)


class VecFrameStack:
    def __init__(self,
                 preproc: atari_utils.AtariPreprocessing,
                 num_frames: int):
        self.preproc = preproc
        self.num_frames = num_frames
        self.frames = collections.deque(maxlen=num_frames)

    def reset(self):
        ob = self.preproc.reset()  # (5, 84, 84, 1)
        for _ in range(self.num_frames):
            self.frames.append(ob)
        return self._get_array()  # (5, 84, 84, 4)

    def step(self, action: np.ndarray):
        # ob.shape = (5, 84, 84, 1)
        # reward.shape = (5,)
        # done.shape = (5,)
        ob, reward, done, info = self.preproc.step(action)
        self.frames.append(ob)
        return self._get_array(), reward, done, info

    def _get_array(self):
        assert len(self.frames) == self.num_frames
        return np.concatenate(self.frames, axis=-1)


def create_env(game: str, clip_rewards: bool, seed: int = None):
    """Create a FrameStack object that serves as environment for the `game`."""
    env = gym.make(game)
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    if clip_rewards:
        env = ClipRewardEnv(env)  # bin rewards to {-1., 0., 1.}
    preproc = atari_utils.AtariPreprocessing(env)
    stack = FrameStack(preproc, num_frames=4)
    return stack


def create_vec_env(game: str, num_envs: int, clip_rewards: bool, seeds: List[int] = None):
    def create_env():
        env = gym.make(game)
        if clip_rewards:
            env = ClipRewardEnv(env)
        preproc = atari_utils.AtariPreprocessing(env)
        return preproc
    env_fns = [create_env] * num_envs
    envs = gym.vector.AsyncVectorEnv(env_fns, shared_memory=True)
    envs.seed(seeds)
    stack = VecFrameStack(envs, num_frames=4)
    return stack


def get_num_actions(env_name: str):
    """Get the number of possible actions of a given Atari game.

    This determines the number of outputs in the actor part of the
    actor-critic model.
    """
    env = gym.make(env_name)
    return env.action_space.n
