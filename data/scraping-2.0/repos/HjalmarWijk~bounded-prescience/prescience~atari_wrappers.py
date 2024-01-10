"""Atari Wrappers from OpenAI Baselines"""

import numpy as np
import os

os.environ.setdefault('PATH', '')
from PIL import Image
from collections import deque
import gym
from gym import spaces
import cv2
import tensorflow as tf
import functools

from prescience.labelling import get_property

cv2.ocl.setUseOpenCL(False)
SEED = 0


class SaveRestoreWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def save(self):
        return self.env.save()

    def _restore(self, savelist, i):
        self.env._restore(savelist, i)

    def restore(self, savelist):
        self._restore(savelist, len(savelist) - 1)


class SaveRestoreObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def save(self):
        return self.env.save()

    def _restore(self, savelist, i):
        self.env._restore(savelist, i)

    def restore(self, savelist):
        self._restore(savelist, len(savelist) - 1)


class SaveRestoreRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def save(self):
        return self.env.save()

    def _restore(self, savelist, i):
        self.env._restore(savelist, i)

    def restore(self, savelist):
        self._restore(savelist, len(savelist) - 1)


class LabellingEnv(SaveRestoreWrapper):
    """Adds a label from a labeller to the info returned at each step"""

    def __init__(self, env, labeller):
        super().__init__(env)
        self.env = env
        self.labeller = labeller

    def reset(self):
        self.labeller.reset()
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['label'] = self.labeller.label(obs, reward, done, info)
        return obs, reward, done, info

    def save(self):
        return self.env.save() + [self.labeller.save()]

    def _restore(self, savelist, i):
        self.labeller.restore(savelist[i])
        self.env._restore(savelist, i - 1)


class FireResetEnv(SaveRestoreWrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        super().__init__(env)
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


class ClipRewardEnv(SaveRestoreRewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(SaveRestoreObsWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None, channel_order='hwc',
                 resize_style='baseline'):
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
        self.resize_style = resize_style
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3
        shape = {
            'hwc': (self._height, self._width, num_colors),
            'chw': (num_colors, self._height, self._width),
        }
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shape[channel_order],
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

        if self.resize_style == 'tf':
            self.inp_shape = [None] + list(env.observation_space.shape[:2]) + [1, ]
            self.x_t = tf.placeholder(tf.float32, self.inp_shape, name='warp_ph')
            self.transform_op = self._transform(self.x_t)

    def _transform(self, obs):
        obs = tf.image.resize_bilinear(obs, (self._width, self._height), align_corners=True)
        obs = tf.reshape(obs, (self._width, self._height) + (1,))
        return obs

    def observation(self, obs):
        if self.resize_style == 'baseline':
            frame = obs
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
            return obs.reshape(self.observation_space.low.shape)
        if self.resize_style == 'tf':
            frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
            frame = frame[np.newaxis, :]
            frame = frame[..., np.newaxis]

            return self.transform_op.eval({self.x_t: frame})
        if self.resize_style == 'np':
            frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
            frame = np.array(Image.fromarray(frame).resize((self._width, self._height),
                                                           resample=Image.BILINEAR), dtype=np.uint8)
            return frame.reshape((self._width, self._height, 1))


class FrameStack(SaveRestoreWrapper):
    def __init__(self, env, k, channel_order='hwc'):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k * 4)
        self.stack_axis = {'hwc': 2, 'chw': 0}[channel_order]
        orig_obs_space = env.observation_space
        low = np.repeat(orig_obs_space.low, k, axis=self.stack_axis)
        high = np.repeat(orig_obs_space.high, k, axis=self.stack_axis)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=orig_obs_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k * 4):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k * 4
        return LazyFrames(self.max_and_skip(), stack_axis=self.stack_axis)

    def max_and_skip(self):
        self.max_skip_frames = [None] * 4
        for i in range(4):
            self.max_skip_frames[i] = np.maximum(self.frames[4 * i + 2], self.frames[4 * i + 3])
        return self.max_skip_frames

    def save(self):
        return self.env.save() + [self.frames]

    def _restore(self, savelist, i):
        self.frames = savelist[i]
        self.env._restore(savelist, i - 1)


class SaveRestoreEnv(SaveRestoreWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.aleenv = env.unwrapped

    def save(self):
        return [self.aleenv.clone_full_state()]

    def _restore(self, savelist, i):
        self.aleenv.restore_full_state(savelist[i])


class GreyscaleEnv(SaveRestoreObsWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def observation(self, observation):
        self.env.ale.getScreenGrayscale(observation)
        return observation


class ScaledFloatFrame(SaveRestoreObsWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames, stack_axis=2):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None
        self.stack_axis = stack_axis

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=self.stack_axis)
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
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


def make_base(env_id, full_action_space=True):
    env = gym.make(env_id, full_action_space=full_action_space)
    env = env.unwrapped  # remove default timelimit
    env.seed(SEED)
    assert 'NoFrameskip' in env.spec.id
    env = SaveRestoreEnv(env)
    return env


def wrap_deepmind(env, labeller, episode_life=False, clip_rewards=False, frame_stack=True, scale=False,
                  channel_order='hwc', resize_style='baseline', standard_greyscale=True):
    """Configure environment for DeepMind-style Atari.
    """
    env = LabellingEnv(env, labeller)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if not standard_greyscale:
        env = GreyscaleEnv(env)
    env = WarpFrame(env, channel_order=channel_order, resize_style=resize_style, grayscale=standard_greyscale)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4, channel_order=channel_order)
    return env


def get_wrapped(env_id, method, label):
    if method in ["Rainbow", "IQN", "DQN-C", "A3C", "PPO", "ACER", "C51"]:
        channel_order = 'chw'
        resize_style = 'baseline'
        scale = False
    if method in ['IMPALA-U']:
        channel_order = 'hwc'
        resize_style = 'np'
        scale = True
    if method in ['A2C', 'APEX']:
        channel_order = 'hwc'
        resize_style = 'tf'
        scale = True
    if method in ['DQN-D', 'Rainbow-D']:
        channel_order = 'hwc'
        resize_style = 'baseline'
        scale = True
    env = make_base(env_id, full_action_space=False)
    labeller = get_property(env, label)
    env = wrap_deepmind(env, labeller, channel_order=channel_order, resize_style=resize_style, scale=scale)
    return env
