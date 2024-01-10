# The MIT License

# Copyright (c) 2017 OpenAI (http://openai.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Code adapted from openAI baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
# Made changes to adapting coding styles, also add new function to support recording video.
#
# ==============================================================================
"""gym environment processing components."""

# Temporally suppress DeprecationWarning
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

from collections import deque
import gym
import copy
from gym.spaces import Box
import numpy as np
import cv2
import datetime


def unwrap(env):
    if hasattr(env, 'unwrapped'):
        return env.unwrapped
    elif hasattr(env, 'env'):
        return unwrap(env.env)
    elif hasattr(env, 'leg_env'):
        return unwrap(env.leg_env)
    else:
        return env


class NoopReset(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """

    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class LifeLossWrapper(gym.Wrapper):
    """Adds boolean key `loss_life` to the info dict, but only reset on true game over."""

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_terminated = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_terminated = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            info['loss_life'] = True
        else:
            info['loss_life'] = False
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_terminated:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkip(gym.Wrapper):
    """Return only every `skip`-th frame"""

    def __init__(self, env, skip=4):
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
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ResizeAndGrayscaleFrame(gym.ObservationWrapper):
    """
    Resize frames to 84x84, and grascale image as done in the Nature paper.
    """

    def __init__(self, env, width=84, height=84, grayscale=True):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )

        original_space = self.observation_space
        self.observation_space = new_space

        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, observation):
        frame = observation

        # pylint: disable=no-member
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        # pylint: disable=no-member

        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        return frame


class FrameStack(gym.Wrapper):
    """Stack k last frames.
    Returns lazy array, which is much more memory efficient.
    See Also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shape = env.observation_space.shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(shape[:-1] + (shape[-1] * k,)),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    """This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    This object should only be converted to numpy array before being passed to the model.
    You'd not believe how complex the previous solution was."""

    def __init__(self, frames):
        self.dtype = frames[0].dtype
        self.shape = (frames[0].shape[0], frames[0].shape[1], len(frames))
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
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


class ClipRewardWithBound(gym.RewardWrapper):
    'Clip reward to in the range [-bound, bound]'

    def __init__(self, env, bound):
        super().__init__(env)
        self._bound = bound

    def reward(self, reward):
        return np.clip(reward, -self._bound, self._bound)


class ObservationChannelFirst(gym.ObservationWrapper):
    """Make observation image channel first, this is for PyTorch only."""

    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        _low, _high = (0.0, 255)
        self.observation_space = Box(
            low=_low,
            high=_high,
            shape=new_shape,
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        # permute [H, W, C] array to  [C, H, W]
        # return np.transpose(observation, axes=(2, 0, 1)).astype(self.observation_space.dtype)
        obs = np.asarray(observation, dtype=self.observation_space.dtype).transpose(2, 0, 1)
        # make sure it's C-contiguous for compress state
        return np.ascontiguousarray(obs, dtype=self.observation_space.dtype)


class ObservationToNumpy(gym.ObservationWrapper):
    """Make the observation into numpy ndarrays."""

    def observation(self, observation):
        return np.asarray(observation, dtype=self.observation_space.dtype)


class ClipObservationWithBound(gym.ObservationWrapper):
    """Make the observation into [-max_abs_value, max_abs_value]."""

    def __init__(self, env, max_abs_value):
        super().__init__(env)
        self._max_abs_value = max_abs_value

    def observation(self, observation):
        return np.clip(observation, -self._max_abs_value, self._max_abs_value)


class RecordRawReward(gym.Wrapper):
    """This wrapper will add non-clipped/unscaled raw reward to the info dict."""

    def step(self, action):
        """Take action and add non-clipped/unscaled raw reward to the info dict."""

        obs, reward, done, info = self.env.step(action)
        info['raw_reward'] = reward

        return obs, reward, done, info


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action

        self.last_action = action
        return self.env.step(action)

    def reset(self, **kwargs):
        self.last_action = 0
        return self.env.reset(**kwargs)


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode'].update(visited_rooms=copy.copy(self.visited_rooms))
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()


def create_atari_environment(
    env_name: str,
    seed: int = 1,
    frame_skip: int = 4,
    frame_stack: int = 4,
    frame_height: int = 84,
    frame_width: int = 84,
    max_noop_steps: int = 30,
    max_episode_steps: int = 108000,
    terminal_on_life_loss: bool = False,
    clip_reward: bool = True,
    grayscale: bool = True,
    sticky_action: bool = True,
    channel_first: bool = True,
) -> gym.Env:
    """
    Process gym env for Atari games according to the Nature DQN paper.

    Args:
        env_name: the environment name without 'NoFrameskip' and version.
        seed: seed the runtime.
        frame_skip: the frequency at which the agent experiences the game,
                the environment will also repeat action.
        frame_stack: stack n last frames.
        frame_height: height of the resized frame.
        frame_width: width of the resized frame.
        max_noop_steps: maximum number of no-ops to apply at the beginning
                of each episode to reduce determinism. These no-ops are applied at a
                low-level, before frame skipping.
        max_episode_steps: maximum steps for an episode.
        terminal_on_life_loss: if True, adds boolean property `loss_life` to the info dict, default off.
        clip_reward: clip reward in the range of [-1, 1], default on.
        grayscale: if True, use grascale image instead of RGB image, default on.
        channel_first: if True, change observation image from shape [H, W, C] to in the range [C, H, W], this is for PyTorch only, default on.

    Returns:
        preprocessed gym.Env for Atari games.
    """
    if 'NoFrameskip' in env_name:
        raise ValueError(f'Environment name should not include NoFrameskip, got {env_name}')

    env = gym.make(f'{env_name}NoFrameskip-v4')
    env.seed(seed)

    # Change TimeLimit wrapper to 108,000 steps (30 min) as default in the
    # literature instead of OpenAI Gym's default of 100,000 steps.
    env = gym.wrappers.TimeLimit(
        env.env,
        max_episode_steps=None if max_episode_steps <= 0 else max_episode_steps,
    )

    if max_noop_steps > 0:
        env = NoopReset(env, noop_max=max_noop_steps)

    if sticky_action:
        env = StickyActionEnv(env)

    if frame_skip > 0:
        env = MaxAndSkip(env, skip=frame_skip)

    if terminal_on_life_loss:
        env = LifeLossWrapper(env)

    env = ResizeAndGrayscaleFrame(env, width=frame_width, height=frame_height, grayscale=grayscale)

    if clip_reward:
        env = RecordRawReward(env)
        env = ClipRewardWithBound(env, 1.0)

    if frame_stack > 1:
        env = FrameStack(env, frame_stack)

    if channel_first:
        env = ObservationChannelFirst(env)
    else:
        # This is required as LazeFrame object is not numpy.array.
        env = ObservationToNumpy(env)

    if 'Montezuma' in env_name or 'Pitfall' in env_name:
        env = MontezumaInfoWrapper(env, room_address=3 if 'Montezuma' in env_name else 1)

    return env


def play_and_record_video(
    agent,
    env: gym.Env,
    save_dir: str = 'recordings',
) -> None:
    """Self-play and record a video for a single game.

    Args:
        agent: the evaluation agent.
        env: the gym environment to play.
        save_dir: the recording video file directory,
            default save to './recordings/{env.spec.id}_{timestamp}'.

    """

    # Create a sub folder with name env.id + timestamp
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    full_save_dir = f'{save_dir}/{env.spec.id}_{ts}'
    print(f'Recording self-play video at "{full_save_dir}"')

    env = gym.wrappers.RecordVideo(env, full_save_dir)

    observation = env.reset()
    done = False

    while True:
        a_t = agent.act(observation)
        observation, reward, done, info = env.step(a_t)

        if done:
            break

    env.close()
