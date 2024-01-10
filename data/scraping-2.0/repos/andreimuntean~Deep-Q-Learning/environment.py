"""Augments OpenAI Gym Atari environments with features like experience replay.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013) and 'Human-level control through deep reinforcement learning' (Mnih et al.,
2015).
"""

import gym
import numpy as np
import random
import time

from gym import wrappers
from scipy import misc

# Specifies restricted action spaces. For games not in this dictionary, all actions are enabled.
ACTION_SPACE = {'Pong-v0': [0, 2, 3],  # NONE, UP and DOWN.
                'Breakout-v0': [1, 2, 3]}  # FIRE (respawn ball, otherwise NOOP), UP and DOWN.


def _preprocess_observation(observation):
    """Transforms the specified observation into a 48x48 grayscale image.

    Returns:
        A 48x48x1 tensor with float16 values between 0 and 1.
    """

    # Transform the observation into a grayscale image with values between 0 and 1. Use the simple
    # np.mean method instead of sophisticated luminance extraction techniques since they do not seem
    # to improve training.
    grayscale_observation = observation.mean(2)

    # Resize grayscale frame to a 48x48 matrix of 16-bit floats.
    resized_observation = misc.imresize(grayscale_observation, (48, 48)).astype(np.float16)

    return resized_observation


class AtariWrapper:
    """Wraps over an Atari environment from OpenAI Gym and provides experience replay."""

    def __init__(self,
                 env_name,
                 max_episode_length,
                 replay_memory_capacity,
                 observations_per_state,
                 action_space=None,
                 save_path=None):
        """Creates the wrapper.

        Args:
            env_name: Name of an OpenAI Gym Atari environment.
            max_episode_length: Maximum number of time steps per episode. When this number of time
                steps is reached, the episode terminates early.
            replay_memory_capacity: Number of experiences remembered. Conceptually, an experience is
                a (state, action, reward, next_state, done) tuple. The replay memory is sampled by
                the agent during training.
            observations_per_state: Number of consecutive observations within a state. Provides some
                short-term memory for the learner. Useful in games like Pong where the trajectory of
                the ball can't be inferred from a single image.
            action_space: A list of possible actions. If 'action_space' is 'None' and no default
                configuration exists for this environment, all actions will be allowed.
            save_path: Path where to save experiments and videos.
        """

        self.gym_env = gym.make(env_name)

        if save_path:
            self.gym_env = wrappers.Monitor(self.gym_env, save_path)

        self.max_episode_length = max_episode_length
        self.replay_memory_capacity = replay_memory_capacity
        self.state_length = observations_per_state
        self.state_shape = [48, 48, observations_per_state]
        self.reset()

        if action_space:
            self.action_space = list(action_space)
        elif env_name in ACTION_SPACE:
            self.action_space = ACTION_SPACE[env_name]
        else:
            self.action_space = list(range(self.gym_env.action_space.n))

        self.num_actions = len(self.action_space)

        # Create replay memory. Arrays are used instead of double-ended queues for faster indexing.
        self.num_exp = 0
        self.actions = np.empty(replay_memory_capacity, np.uint8)
        self.rewards = np.empty(replay_memory_capacity, np.int8)
        self.ongoing = np.empty(replay_memory_capacity, np.bool)

        # Used for computing both 'current' and 'next' states.
        self.observations = np.empty([replay_memory_capacity + observations_per_state, 48, 48],
                                     np.float16)

        # Initialize the first state by performing random actions.
        for i in range(observations_per_state):
            observation, _, _, _ = self.gym_env.step(self.sample_action())
            self.observations[i] = _preprocess_observation(observation)

        # Initialize the first experience by performing one more random action.
        self.step(self.sample_action())

    def reset(self):
        """Resets the environment."""

        self.done = False
        self.gym_env.reset()
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_start_time = time.time()
        self.episode_run_time = 0
        self.fps = 0
        self.lives = None

    def step(self, action):
        """Performs the specified action.

        Returns:
            The reward.

        Raises:
            ValueError: If the action is not valid.
        """

        if self.done:
            self.reset()

        if action not in self.action_space:
            raise ValueError('Action "{}" is invalid. Valid actions: {}.'.format(action,
                                                                                 self.action_space))

        observation, reward, self.done, info = self.gym_env.step(action)
        observation = _preprocess_observation(observation)

        self.episode_reward += reward
        self.episode_length += 1
        self.episode_run_time = time.time() - self.episode_start_time
        self.fps = 0 if self.episode_run_time == 0 else self.episode_length / self.episode_run_time

        if self.episode_length == self.max_episode_length:
            self.done = True

        # Treat loss of life as end of episode.
        ongoing = not self.done and self.lives is not None and self.lives == info['ale.lives']
        self.lives = info['ale.lives']

        # Remember this experience.
        self.actions[self.num_exp] = action
        self.rewards[self.num_exp] = -1 if reward < 0 else 1 if reward > 0 else 0
        self.ongoing[self.num_exp] = ongoing
        self.observations[self.num_exp + self.state_length] = observation
        self.num_exp += 1

        if self.num_exp == self.replay_memory_capacity:
            # Free up space by deleting half of the oldest experiences.
            mid = int(self.num_exp / 2)
            end = 2 * mid

            self.num_exp = mid
            self.actions[:mid] = self.actions[mid:end]
            self.rewards[:mid] = self.rewards[mid:end]
            self.ongoing[:mid] = self.ongoing[mid:end]
            self.observations[:mid + self.state_length] = self.observations[mid:
                                                                            end + self.state_length]

        return reward

    def render(self):
        """Draws the environment."""

        self.gym_env.render()

    def sample_action(self):
        """Samples a random action."""

        return random.choice(self.action_space)

    def sample_experiences(self, exp_count):
        """Randomly samples experiences from the replay memory. May contain duplicates.

        Args:
            exp_count: Number of experiences to sample.

        Returns:
            A (states, actions, rewards, next_states, ongoing) tuple. The boolean array, 'ongoing',
            determines whether the 'next_states' are terminal states.
        """

        indexes = np.random.choice(self.num_exp, exp_count)
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        ongoing = self.ongoing[indexes]
        states = np.array([self._get_state(i) for i in indexes])
        next_states = np.array([self._get_state(i) for i in indexes + 1])

        return states, actions, rewards, next_states, ongoing

    def get_state(self):
        """Gets the current state.

        Returns:
            A 1x48x48x(self.observations_per_state) tensor with float16 values between 0 and 1.
        """

        return np.expand_dims(self._get_state(-1), axis=0)

    def _get_state(self, index):
        """Gets the specified state. Supports negative indexing.

        States are series of consecutive observations. Using more than one observation per state may
        provide short-term memory for the learner. Great for games like Pong where the trajectory of
        the ball can't be inferred from a single image.

        Returns:
            A 48x48x(observations_per_state) tensor with float16 values between 0 and 1.
        """

        state = np.empty([48, 48, self.state_length], np.float16)

        # Allow negative indexing by wrapping around.
        index = index % (self.num_exp + 1)

        for i in range(self.state_length):
            state[..., i] = self.observations[index + i]

        return state
