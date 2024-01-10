# -*- coding: utf-8 -*-

import os, sys
import random
import json
import copy
import re
import time
from tqdm import tqdm

from collections import deque

import numpy as np

import cv2

import tensorflow as tf
from tensorflow import orthogonal_initializer, constant_initializer

import scipy.signal

import gym
from gym import spaces

import threading

# ----------
# REFERENCE
# https://github.com/imai-laboratory/dqn
# https://github.com/imai-laboratory/rlsaber/tree/master/rlsaber


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# rlsaber.explorer.py
# -----------------------------------------------------------------------------------------------------------

class ConstantExplorer:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def select_action(self, t, greedy_action, num_actions):
        if random.random() < self.epsilon:
            return np.random.choice(num_actions)
        return greedy_action

class LinearDecayExplorer:
    def __init__(self, final_exploration_step=10**6,
                start_epsilon=1.0, final_epsilon=0.1):
        self.final_exploration_step = final_exploration_step
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.base_epsilon = self.start_epsilon - self.final_epsilon

    def select_action(self, t, greedy_action, num_actions):
        factor = 1 - float(t) / self.final_exploration_step
        if factor < 0:
            factor = 0
        eps = self.base_epsilon * factor + self.final_epsilon
        if random.random() < eps:
            return np.random.choice(num_actions)
        return greedy_action


# -----------------------------------------------------------------------------------------------------------
# rlsaber.log.py
# -----------------------------------------------------------------------------------------------------------

# dummy class to restore constant file
class Constant:
    pass


# dump constant variables into json file
def dump_constants(constants, path):
    data = {}
    for name in dir(constants):
        if re.match(r'^([A-Z]|_|[0-9])+$', name):
            data[name] = getattr(constants, name)
    json_str = json.dumps(data)
    with open(path, 'w') as f:
        f.write(json_str + '\n')


# restore a constant object from json file
def restore_constants(path):
    constants = Constant()
    with open(path, 'r') as f:
        json_obj = json.loads(f.read())
        for key, value in json_obj.items():
            setattr(constants, key, value)
    return constants


class TfBoardLogger:
    def __init__(self, writer):
        self.placeholders = {}
        self.summaries = {}
        self.writer = writer

    def register(self, name, dtype):
        placeholder = tf.placeholder(dtype, [], name=name)
        self.placeholders[name] = placeholder
        self.summaries[name] = tf.summary.scalar(name + '_summary', placeholder)

    def register_image(self, name, shape, n):
        placeholder = tf.placeholder(tf.float32, [None] + shape, name=name)
        self.placeholders[name] = placeholder
        self.summaries[name] = tf.summary.image(name + '_summary', placeholder, n)

    def plot(self, name, value, step):
        sess = tf.get_default_session()
        placeholder = self.placeholders[name]
        summary = self.summaries[name]
        out, _ = sess.run(
            [summary, placeholder],
            feed_dict={placeholder: value}
        )
        self.writer.add_summary(out, step)

class JsonLogger:
    def __init__(self, path, overwrite=True):
        self.f = open(path, 'w' if overwrite else 'wa')

    def plot(self, **kwargs):
        json_str = json.dumps(kwargs)
        self.f.write(json_str + '\n')

    def close(self):
        self.f.close()


# -----------------------------------------------------------------------------------------------------------
# rlsaber.preprocess.py
# -----------------------------------------------------------------------------------------------------------

def atari_preprocess(image, shape):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(gray, (210, 160))
    state = cv2.resize(state, (84, 110))
    state = state[18:102, :]
    state = cv2.resize(state, tuple(shape))
    return state


# -----------------------------------------------------------------------------------------------------------
# rlsaber.replay_buffer.py
# -----------------------------------------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def append(self, obs_t, action, reward, obs_tp1, done):
        if isinstance(done, bool):
            done = 1 if done else 0
        experience = dict(obs_t=obs_t, action=action,
                reward=reward, obs_tp1=obs_tp1, done=done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        obs_t = []
        actions = []
        rewards = []
        obs_tp1 = []
        done = []
        for experience in experiences:
            obs_t.append(experience['obs_t'])
            actions.append(experience['action'])
            rewards.append(experience['reward'])
            obs_tp1.append(experience['obs_tp1'])
            done.append(experience['done'])
        return obs_t, actions, rewards, obs_tp1, done

class EpisodeReplayBuffer:
    def __init__(self, episode_size):
        self.buffer = deque(maxlen=episode_size)
        self.tmp_obs_t_buffer = []
        self.tmp_action_buffer = []
        self.tmp_reward_buffer = []
        self.tmp_obs_tp1_buffer = []
        self.tmp_done_buffer = []

    def append(self, obs_t, action, reward, obs_tp1, done):
        if isinstance(done, bool):
            done = 1 if done else 0
        self.tmp_obs_t_buffer.append(obs_t)
        self.tmp_action_buffer.append(action)
        self.tmp_reward_buffer.append(reward)
        self.tmp_obs_tp1_buffer.append(obs_tp1)
        self.tmp_done_buffer.append(done)

    def end_episode(self):
        episode = dict(
            obs_t=self.tmp_obs_t_buffer,
            action=self.tmp_action_buffer,
            reward=self.tmp_reward_buffer,
            obs_tp1=self.tmp_obs_tp1_buffer,
            done=self.tmp_done_buffer
        )
        self.buffer.append(episode)
        self.reset_tmp_buffer()

    def reset_tmp_buffer(self):
        self.tmp_obs_t_buffer = []
        self.tmp_action_buffer = []
        self.tmp_reward_buffer = []
        self.tmp_obs_tp1_buffer = []
        self.tmp_done_buffer = []

    def sample_episodes(self, batch_size):
        episodes = random.sample(self.buffer, batch_size)
        obs_t = []
        actions = []
        rewards = []
        obs_tp1 = []
        done = []
        for episode in episodes:
            obs_t.append(episode['obs_t'])
            actions.append(episode['action'])
            rewards.append(episode['reward'])
            obs_tp1.append(episode['obs_tp1'])
            done.append(episode['done'])
        return obs_t, actions, rewards, obs_tp1, done

    def sample_sequences(self, batch_size, step_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        obs_t = []
        actions = []
        rewards = []
        obs_tp1 = []
        done = []
        for index in indices:
            episode = self.buffer[index]
            start_pos = np.random.randint(len(episode['obs_t']) - step_size + 1)
            obs_t.append(episode['obs_t'][start_pos:start_pos+step_size])
            actions.append(episode['action'][start_pos:start_pos+step_size])
            rewards.append(episode['reward'][start_pos:start_pos+step_size])
            obs_tp1.append(episode['obs_tp1'][start_pos:start_pos+step_size])
            done.append(episode['done'][start_pos:start_pos+step_size])
        return obs_t, actions, rewards, obs_tp1, done

class NECReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def append(self, obs_t, action, value):
        experience = dict(obs_t=obs_t, action=action, value=value)
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        obs_t = []
        actions = []
        values = []
        for experience in experiences:
            obs_t.append(experience['obs_t'])
            actions.append(experience['action'])
            values.append(experience['value'])
        return obs_t, actions, values


# -----------------------------------------------------------------------------------------------------------
# rlsaber.tf_util.py
# -----------------------------------------------------------------------------------------------------------

# from OpenAI baseline
# https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py
def lstm(xs, ms, s, nh, init_scale=1.0):
    nbatch, nin = [v.value for v in xs[0].get_shape()]
    wx = tf.get_variable("wx", [nin, nh*4], initializer=orthogonal_initializer(init_scale))
    wh = tf.get_variable("wh", [nh, nh*4], initializer=orthogonal_initializer(init_scale))
    b = tf.get_variable("b", [nh*4], initializer=constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, (x, m) in enumerate(zip(xs, ms)):
        c = c*(1-m)
        h = h*(1-m)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s

# (batch_size * step_size, dim) -> (batch_size, step_size, dim) -> (step_size, batch_size, dim)
def batch_to_seq(batch, batch_size, step_size):
    seq = tf.reshape(batch, [batch_size, step_size, int(batch.shape[1])])
    seq = [tf.squeeze(v, axis=1) for v in tf.split(seq, num_or_size_splits=step_size, axis=1)]
    return seq


# (step_size, batch_size, dim) -> (batch_size, step_size, dim) -> (batch_size * step_size, dim)
def seq_to_batch(seq, batch_size, step_size):
    seq = tf.concat(seq, axis=1)
    seq = tf.reshape(seq, [step_size, batch_size, -1])
    seq = tf.transpose(seq, [1, 0, 2])
    batch = tf.reshape(seq, [batch_size * step_size, -1])
    return batch


# -----------------------------------------------------------------------------------------------------------
# rlsaber.util.py
# -----------------------------------------------------------------------------------------------------------

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def compute_v_and_adv(rewards, values, bootstrapped_value, gamma, lam=1.0):
    rewards = np.array(rewards)
    values = np.array(list(values) + [bootstrapped_value])
    v = discount(np.array(list(rewards) + [bootstrapped_value]), gamma)[:-1]
    delta = rewards + gamma * values[1:] - values[:-1]
    adv = discount(delta, gamma * lam)
    return v, adv

def compute_returns(rewards, bootstrap_value, terminals, gamma):
    # (N, T) -> (T, N)
    rewards = np.transpose(rewards, [1, 0])
    terminals = np.transpose(terminals, [1, 0])
    returns = []
    R = bootstrap_value
    for i in reversed(range(rewards.shape[0])):
        R = rewards[i] + (1.0 - terminals[i]) * gamma * R
        returns.append(R)
    returns = reversed(returns)
    # (T, N) -> (N, T)
    returns = np.transpose(list(returns), [1, 0])
    return returns

def compute_gae(rewards, values, bootstrap_values, terminals, gamma, lam):
    # (N, T) -> (T, N)
    rewards = np.transpose(rewards, [1, 0])
    values = np.transpose(values, [1, 0])
    values = np.vstack((values, [bootstrap_values]))
    terminals = np.transpose(terminals, [1, 0])
    # compute delta
    deltas = []
    for i in reversed(range(rewards.shape[0])):
        V = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))
    # compute gae
    A = deltas[-1,:]
    advantages = [A]
    for i in reversed(range(deltas.shape[0] - 1)):
        A = deltas[i] + (1.0 - terminals[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)
    # (T, N) -> (N, T)
    advantages = np.transpose(list(advantages), [1, 0])
    return advantages

class Rollout:
    def __init__(self):
        self.flush()

    def add(self, state, action, reward, value, terminal=False, feature=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.terminals.append(terminal)
        self.features.append(feature)

    def flush(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.terminals = []
        self.features = []


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# rlsaber.env.env_wrapper.py
# -----------------------------------------------------------------------------------------------------------

class EnvWrapper:
    def __init__(self, env, r_preprocess=None, s_preprocess=None):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.r_preprocess = r_preprocess
        self.s_preprocess = s_preprocess
        self.results = {
            'rewards': 0
        }

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.results['rewards'] += reward
        # preprocess reward
        if self.r_preprocess is not None:
            reward = self.r_preprocess(reward)
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        # preprocess state
        if self.s_preprocess is not None:
            state = self.s_preprocess(state)
        self.results['rewards'] = 0
        return state

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def get_results(self):
        return self.results

class ActionRepeatEnvWrapper(EnvWrapper):
    def __init__(self, env, r_preprocess=None, s_preprocess=None, repeat=4):
        super().__init__(env, r_preprocess, s_preprocess)
        self.repeat = repeat
        self.states = deque(maxlen=repeat)

    def step(self, action):
        sum_of_reward = 0
        done = False
        for i in range(self.repeat):
            if done:
                state = np.zeros_like(np.array(state))
                reward = 0
            else:
                state, reward, done, info = super().step(action)
            sum_of_reward += reward
            self.states.append(state)
        return np.array(list(self.states)), sum_of_reward, done, info

    def reset(self):
        state = super().reset()
        for i in range(self.repeat):
            init_state = np.zeros_like(np.array(state))
            self.states.append(init_state)
        self.states.append(state)
        return np.array(list(self.states))

class BatchEnvWrapper:
    def __init__(self, envs):
        self.envs = envs
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        state_shape = envs[0].observation_space.shape
        self.zero_state = np.zeros_like(self.reset(0), dtype=np.float32)
        self.results = [env.get_results() for env in self.envs]

    def step(self, actions):
        states = []
        rewards = []
        dones = []
        infos = []
        for i, env in enumerate(self.envs):
            state, reward, done, info = env.step(actions[i])
            self.results[i] = copy.copy(env.get_results())
            if done:
                state = env.reset()
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return states, rewards, dones, infos

    def reset(self, index):
        return self.envs[index].reset()

    def render(self, mode='human'):
        return self.envs[0].render(mode=mode)

    def get_num_of_envs(self):
        return len(self.envs)

    def get_results(self):
        return self.results

# from https://github.com/openai/baselines
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

# from https://github.com/openai/baselines
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

# from https://github.com/openai/baselines
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

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

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# -----------------------------------------------------------------------------------------------------------
# rlsaber.env.mario.py
# -----------------------------------------------------------------------------------------------------------

# import gym_pull
# from gym.spaces import Discrete
#
# gym_pull.pull('github.com/ppaquette/gym-super-mario')
#
# from ppaquette_gym_super_mario import wrappers
#
# # manual controller mapping to discrete actions
# # beucase gym-super-mario uses DiscreteToMultiDiscrete
# # which is removed in the latest gym
# action_mapping = {
#     0:  [0, 0, 0, 0, 0, 0],  # NOOP
#     1:  [1, 0, 0, 0, 0, 0],  # Up
#     2:  [0, 0, 1, 0, 0, 0],  # Down
#     3:  [0, 1, 0, 0, 0, 0],  # Left
#     4:  [0, 1, 0, 0, 1, 0],  # Left + A
#     5:  [0, 1, 0, 0, 0, 1],  # Left + B
#     6:  [0, 1, 0, 0, 1, 1],  # Left + A + B
#     7:  [0, 0, 0, 1, 0, 0],  # Right
#     8:  [0, 0, 0, 1, 1, 0],  # Right + A
#     9:  [0, 0, 0, 1, 0, 1],  # Right + B
#     10: [0, 0, 0, 1, 1, 1],  # Right + A + B
#     11: [0, 0, 0, 0, 1, 0],  # A
#     12: [0, 0, 0, 0, 0, 1],  # B
#     13: [0, 0, 0, 0, 1, 1],  # A + B
# }
#
# modewrapper = wrappers.SetPlayingMode('algo')
#
# class MarioEnv:
#     def __init__(self, env_name):
#         self.env_name = env_name
#         self.env = None
#         self.action_space = Discrete(len(action_mapping.keys()))
#
#     def reset(self):
#         if self.env is not None:
#             self.env.close()
#         self.env = modewrapper(gym.make(self.env_name))
#         return self.env.reset()
#
#     def step(self, action):
#         controller_action = action_mapping[action]
#         return self.env.step(controller_action)
#
# def make(env_name):
#     return MarioEnv(env_name)


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# rlsaber.trainer.evaluator.py
# -----------------------------------------------------------------------------------------------------------

class Evaluator:
    def __init__(self,
                 env,
                 state_shape=[84, 84],
                 state_window=1,
                 eval_episodes=10,
                 render=False,
                 recorder=None,
                 record_episodes=3):
        self.env = env
        self.state_shape = state_shape
        self.eval_episodes = eval_episodes
        self.render = render
        self.recorder = recorder
        self.record_episodes = record_episodes

        self.init_states = deque(
            np.zeros([state_window] + state_shape, dtype=np.float32).tolist(),
            maxlen=state_window)

    def start(self, agent, trainer_step, trainer_episode):
        episode = 0
        rewards = []
        if self.recorder is not None:
            recorded_episodes = np.random.choice(
                self.eval_episodes, self.record_episodes, replace=False)
            recorders = {i: copy.deepcopy(self.recorder) for i in recorded_episodes}
        while True:
            sum_of_rewards = 0
            reward = 0
            done = False
            state = self.env.reset()
            states = copy.deepcopy(self.init_states)
            while True:
                states.append(state.tolist())
                nd_states = np.array(list(states))

                if self.render:
                    self.env.render()

                if self.recorder is not None and episode in recorders:
                    recorders[episode].append(self.env.render(mode='rgb_array'))

                # episode reaches the end
                if done:
                    episode += 1
                    rewards.append(sum_of_rewards)
                    agent.stop_episode(nd_states, reward, False)
                    break

                action = agent.act(nd_states, reward, False)
                state, reward, done, info = self.env.step(action)

                sum_of_rewards += reward

            if episode == self.eval_episodes:
                break

        if self.recorder is not None:
            for index, recorder in recorders.items():
                recorder.save_mp4('{}_{}.mp4'.format(trainer_step, index))
                recorder.flush()

        return rewards

class Recorder:
    def __init__(self, outdir, bgr=True):
        self.outdir = outdir
        self.images = []
        self.bgr = bgr

    def append(self, image):
        self.images.append(image)

    def save_mp4(self, file_name):
        path = os.path.join(self.outdir, file_name)
        save_video(path, self.images, bgr=self.bgr)

    def flush(self):
        self.images = []

def save_video(path, images, frame_rate=30.0, bgr=True):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(path, fourcc, frame_rate, (width, height), True)
    for image in images:
        if bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        writer.write(image)
    writer.release()


# -----------------------------------------------------------------------------------------------------------
# rlsaber.trainer.trainer.py
# -----------------------------------------------------------------------------------------------------------

class AgentInterface:
    def act(self, state, reward, training):
        raise NotImplementedError()

    def stop_episode(state, reward, training):
        raise NotImplementedError()

class Trainer:
    def __init__(self,
                 env,
                 agent,
                 state_shape=[84, 84],
                 final_step=1e7,
                 state_window=1,
                 training=True,
                 render=False,
                 debug=True,
                 progress_bar=True,
                 before_action=None,
                 after_action=None,
                 end_episode=None,
                 is_finished=None,
                 evaluator=None,
                 end_eval=None,
                 should_eval=lambda s, e: s % 10 ** 5 == 0):
        self.env = env
        self.final_step = final_step
        self.init_states = deque(
            np.zeros(
                [state_window] + state_shape,
                dtype=np.float32
            ).tolist(),
            maxlen=state_window
        )
        self.agent = agent
        self.training = training
        self.render = render
        self.debug = debug
        self.progress_bar = progress_bar
        self.before_action = before_action
        self.after_action = after_action
        self.end_episode = end_episode
        self.is_finished = is_finished
        self.evaluator = evaluator
        self.end_eval = end_eval
        self.should_eval = should_eval

        # counters
        self.global_step = 0
        self.local_step = 0
        self.episode = 0
        self.sum_of_rewards = 0
        self.pause = True

        # for multithreading
        self.resume_event = threading.Event()
        self.resume_event.set()

    def move_to_next(self, states, reward, done):
        states = np.array(list(states))
        # take next action
        action = self.agent.act(
            states,
            reward,
            self.training
        )
        state, reward, done, info = self.env.step(action)
        # render environment
        if self.render:
            self.env.render()
        return state, reward, done, info

    def finish_episode(self, states, reward):
        states = np.array(list(states))
        self.agent.stop_episode(
            states,
            reward,
            self.training
        )

    def start(self):
        if self.progress_bar:
            pbar = tqdm(total=self.final_step, dynamic_ncols=True)
        while True:
            self.local_step = 0
            self.sum_of_rewards = 0
            reward = 0
            done = False
            state = self.env.reset()
            states = copy.deepcopy(self.init_states)
            while True:
                # to stop trainer from outside
                self.resume_event.wait()

                states.append(state.tolist())

                # episode reaches the end
                if done:
                    raw_reward = self.env.get_results()['rewards']
                    self.episode += 1
                    if self.progress_bar:
                        pbar.update(self.local_step)
                        msg = 'step: {}, episode: {}, reward: {}'.format(
                            self.global_step, self.episode, raw_reward)
                        pbar.set_description(msg)
                    self.end_episode_callback(
                        raw_reward, self.global_step, self.episode)
                    self.finish_episode(states, reward)
                    break

                self.before_action_callback(
                    states, self.global_step, self.local_step)

                state, reward, done, info = self.move_to_next(
                    states, reward, done)

                self.after_action_callback(
                    states, reward, self.global_step, self.local_step)

                self.sum_of_rewards += reward
                self.global_step += 1
                self.local_step += 1

                if self.evaluator is not None:
                    self.evaluate()

            if self.is_training_finished():
                if self.progress_bar:
                    pbar.close()
                return

    def before_action_callback(self, states, global_step, local_step):
        if self.before_action is not None:
            self.before_action(
                states,
                global_step,
                local_step
            )

    def after_action_callback(self, states, reward, global_step, local_step):
        if self.after_action is not None:
            self.after_action(
                states,
                reward,
                global_step,
                local_step
            )

    def end_episode_callback(self, reward, global_step, episode):
        if self.end_episode is not None:
            self.end_episode(
                reward,
                global_step,
                episode
            )

    def is_training_finished(self):
        if self.is_finished is not None:
            return self.is_finished(self.global_step)
        return self.global_step > self.final_step

    def evaluate(self):
        should_eval = self.should_eval(self.global_step, self.episode)
        if should_eval:
            print('evaluation starts')
            agent = copy.copy(self.agent)
            agent.stop_episode(copy.deepcopy(self.init_states), 0, False)
            eval_rewards = self.evaluator.start(
                agent, self.global_step, self.episode)
            if self.end_eval is not None:
                self.end_eval(self.global_step, self.episode, eval_rewards)
            if self.debug:
                msg = '[eval] step: {}, episode: {}, reward: {}'
                print(msg.format(
                    self.global_step, self.episode, np.mean(eval_rewards)))

    def stop(self):
        self.resume_event.clear()

    def resume(self):
        self.resume_event.set()

class BatchTrainer(Trainer):
    def __init__(self,
                env, # BatchEnvWrapper
                agent,
                state_shape=[84, 84],
                final_step=1e7,
                state_window=1,
                training=True,
                render=False,
                debug=True,
                before_action=None,
                after_action=None,
                end_episode=None):
        super().__init__(
            env=env,
            agent=agent,
            state_shape=state_shape,
            final_step=final_step,
            state_window=state_window,
            training=training,
            render=render,
            debug=debug,
            before_action=before_action,
            after_action=after_action,
            end_episode=end_episode
        )

        # overwrite global_step
        self.global_step = 0

    # TODO: Remove this overwrite
    def move_to_next(self, states, reward, done):
        # take next action
        action = self.agent.act(
            states,
            reward,
            done, # overwrite line this
            self.training
        )
        state, reward, done, info = self.env.step(action)
        # render environment
        if self.render:
            self.env.render()
        return state, reward, done, info

    # overwrite
    def start(self):
        to_ndarray = lambda q: np.array(list(map(lambda s: list(s), copy.deepcopy(q))))

        # values for the number of n environment
        n_envs = self.env.get_num_of_envs()
        self.local_step = [0 for _ in range(n_envs)]
        self.sum_of_rewards = [0 for _ in range(n_envs)]
        rewards = [0 for _ in range(n_envs)]
        dones = [False for _ in range(n_envs)]
        states = [self.env.reset(i) for i in range(n_envs)]
        queue_states = [copy.deepcopy(self.init_states) for _ in range(n_envs)]
        for i, state in enumerate(states):
            queue_states[i].append(state.tolist())
        t = 0
        pbar = tqdm(total=self.final_step, dynamic_ncols=True)

        # training loop
        while True:
            for i in range(n_envs):
                self.before_action_callback(
                    states[i],
                    self.global_step,
                    self.local_step[i]
                )

            # backup episode status
            prev_dones = dones
            states, rewards, dones, infos = self.move_to_next(
                to_ndarray(queue_states), rewards, prev_dones)

            for i in range(n_envs):
                self.after_action_callback(
                    states[i],
                    rewards[i],
                    self.global_step,
                    self.local_step[i]
                )

            # add state to queue
            for i, (state, done) in enumerate(zip(states, dones)):
                if done:
                    raw_reward = self.env.get_results()[i]['rewards']
                    self.episode += 1
                    global_step = self.global_step - (n_envs - i - 1)
                    msg = 'step: {}, episode: {}, reward: {}'
                    pbar.update(self.local_step[i])
                    pbar.set_description(
                        msg.format(global_step, self.episode, raw_reward))
                    # callback at the end of episode
                    self.end_episode(raw_reward, global_step, self.episode)
                    queue_states[i] = copy.deepcopy(self.init_states)
                    self.sum_of_rewards[i] = 0
                    self.local_step[i] = 0
                queue_states[i].append(state)

            for i in range(n_envs):
                self.sum_of_rewards[i] += rewards[i]
                if not dones[i]:
                    self.global_step += 1
                    self.local_step[i] += 1

            t += 1

            if self.is_training_finished():
                pbar.close()
                return

class AsyncTrainer:
    def __init__(self,
                envs,
                agents,
                state_shape=[84, 84],
                final_step=1e7,
                state_window=1,
                training=True,
                render=False,
                debug=True,
                progress_bar=True,
                before_action=None,
                after_action=None,
                end_episode=None,
                n_threads=10,
                evaluator=None,
                end_eval=None,
                should_eval=None):
        # meta data shared by all threads
        self.meta_data = {
            'shared_step': 0,
            'shared_episode': 0,
            'last_eval_step': 0,
            'last_eval_episode': 0
        }
        if progress_bar:
            pbar = tqdm(total=final_step, dynamic_ncols=True)

        # inserted callbacks
        def _before_action(state, global_step, local_step):
            shared_step = self.meta_data['shared_step']
            if before_action is not None:
                before_action(state, shared_step, global_step, local_step)

        def _after_action(state, reward, global_step, local_step):
            self.meta_data['shared_step'] += 1
            shared_step = self.meta_data['shared_step']
            if after_action is not None:
                after_action(state, reward, shared_step, global_step, local_step)

        def _end_episode(i):
            def func(reward, global_step, episode):
                shared_step = self.meta_data['shared_step']
                self.meta_data['shared_episode'] += 1
                shared_episode = self.meta_data['shared_episode']
                if end_episode is not None:
                    end_episode(
                        reward,
                        shared_step,
                        global_step,
                        shared_episode,
                        episode
                    )
                if progress_bar:
                    pbar.update(self.trainers[i].local_step)
                    msg = 'step: {}, episode: {}, reward: {}'.format(
                        shared_step, shared_episode, reward)
                    pbar.set_description(msg)
            return func

        def _end_eval(step, episode, rewards):
            shared_step = self.meta_data['shared_step']
            shared_episode = self.meta_data['shared_episode']
            for trainer in self.trainers:
                trainer.resume()
            if debug:
                msg = '[eval] step: {}, episode: {}, reward: {}'
                print(msg.format(shared_step, shared_episode, np.mean(rewards)))
            if end_eval is not None:
                end_eval(shared_step, shared_episode, step, episode, rewards)

        def _should_eval(step, episode):
            shared_step = self.meta_data['shared_step']
            shared_episode = self.meta_data['shared_episode']
            last_eval_step = self.meta_data['last_eval_step']
            last_eval_episode = self.meta_data['last_eval_episode']
            if should_eval is not None:
                is_eval = should_eval(
                    last_eval_step, last_eval_episode,
                    shared_step, shared_episode, step, episode)
                if is_eval:
                    for trainer in self.trainers:
                        trainer.stop()
                    self.meta_data['last_eval_step'] = shared_step
                    self.meta_data['last_eval_episode'] = shared_episode
            return is_eval

        self.trainers = []
        for i in range(n_threads):
            env = envs[i]
            agent = agents[i]
            trainer = Trainer(
                env=env,
                agent=agent,
                state_shape=state_shape,
                final_step=final_step,
                state_window=state_window,
                training=training,
                render=i == 0 and render,
                debug=False,
                progress_bar=False,
                before_action=_before_action,
                after_action=_after_action,
                end_episode=_end_episode(i),
                is_finished=lambda s: self.meta_data['shared_step'] > final_step,
                evaluator=evaluator if i == 0 else None,
                should_eval=_should_eval if i == 0 else None,
                end_eval=_end_eval if i == 0 else None
            )
            self.trainers.append(trainer)

    def start(self):
        sess = tf.get_default_session()
        coord = tf.train.Coordinator()
        # gym renderer is only available on the main thread
        render_trainer = self.trainers[0]
        threads = []
        for i in range(len(self.trainers) - 1):
            def run(index):
                with sess.as_default():
                    self.trainers[index + 1].start()
            thread = threading.Thread(target=run, args=(i,))
            thread.start()
            threads.append(thread)
            time.sleep(0.1)
        render_trainer.start()
        coord.join(threads)
