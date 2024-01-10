# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for handling the environment."""

import gym
import numpy as np

from .env import Base2048Env


class ClipRewardEnv(gym.RewardWrapper):
    """
    Adapted from OpenAI baselines.
    github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


def create_env(clip_rewards: bool, max_steps: int) -> Base2048Env:
    """Create a FrameStack object that serves as environment for the `game`."""
    env = Base2048Env()
    env = gym.wrappers.TimeLimit(env, max_steps)
    if clip_rewards:
        env = ClipRewardEnv(env)  # bin rewards to {-1., 0., 1.}
    return env  # noqa
