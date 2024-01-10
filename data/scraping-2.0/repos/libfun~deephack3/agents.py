import time
import logging
import numpy as np
import cv2
import go_vncdriver
import tensorflow as tf
import gym
from gym import wrappers
from gym.spaces.box import Box
from universe import vectorized
from universe.wrappers import  Unvectorize, Vectorize

import numpy as np
import tensorflow as tf
import os
import sys
import re
import time
import random
import argparse
import six
import cv2

from tensorpack import *
from tensorpack.RL import *
from common import play_one_episode

logger = logging.getLogger()
logger.setLevel(logging.INFO)

NUM_EPISODES =  5
CHECKPOINT_LOCATION = '/home/ubuntu/pacman/train'
MONITOR_LOCATION = "./pacman-1"

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

NUM_ACTIONS = None
ENV_NAME = None

#
# This was copy-pasted from openai/universe-starter-agent
#
def DiagnosticsInfo(env, *args, **kwargs):
    return vectorized.VectorizeFilter(env, DiagnosticsInfoI, *args, **kwargs)

class DiagnosticsInfoI(vectorized.Filter):
    def __init__(self, log_interval=503):
        super(DiagnosticsInfoI, self).__init__()

        self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = log_interval
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        self._num_vnc_updates = 0
        self._last_episode_id = -1

    def _after_reset(self, observation):
        logger.info('Resetting environment')
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        return observation

    def _after_step(self, observation, reward, done, info):
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()

        self._local_t += 1
        if info.get("stats.vnc.updates.n") is not None:
            self._num_vnc_updates += info.get("stats.vnc.updates.n")

        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = cur_time - self._last_time
            fps = self._log_interval / elapsed
            self._last_time = cur_time
            cur_episode_id = info.get('vectorized.episode_id', 0)
            to_log["diagnostics/fps"] = fps
            if self._last_episode_id == cur_episode_id:
                to_log["diagnostics/fps_within_episode"] = fps
            self._last_episode_id = cur_episode_id
            if info.get("stats.gauges.diagnostics.lag.action") is not None:
                to_log["diagnostics/action_lag_lb"] = info["stats.gauges.diagnostics.lag.action"][0]
                to_log["diagnostics/action_lag_ub"] = info["stats.gauges.diagnostics.lag.action"][1]
            if info.get("reward.count") is not None:
                to_log["diagnostics/reward_count"] = info["reward.count"]
            if info.get("stats.gauges.diagnostics.clock_skew") is not None:
                to_log["diagnostics/clock_skew_lb"] = info["stats.gauges.diagnostics.clock_skew"][0]
                to_log["diagnostics/clock_skew_ub"] = info["stats.gauges.diagnostics.clock_skew"][1]
            if info.get("stats.gauges.diagnostics.lag.observation") is not None:
                to_log["diagnostics/observation_lag_lb"] = info["stats.gauges.diagnostics.lag.observation"][0]
                to_log["diagnostics/observation_lag_ub"] = info["stats.gauges.diagnostics.lag.observation"][1]

            if info.get("stats.vnc.updates.n") is not None:
                to_log["diagnostics/vnc_updates_n"] = info["stats.vnc.updates.n"]
                to_log["diagnostics/vnc_updates_n_ps"] = self._num_vnc_updates / elapsed
                self._num_vnc_updates = 0
            if info.get("stats.vnc.updates.bytes") is not None:
                to_log["diagnostics/vnc_updates_bytes"] = info["stats.vnc.updates.bytes"]
            if info.get("stats.vnc.updates.pixels") is not None:
                to_log["diagnostics/vnc_updates_pixels"] = info["stats.vnc.updates.pixels"]
            if info.get("stats.vnc.updates.rectangles") is not None:
                to_log["diagnostics/vnc_updates_rectangles"] = info["stats.vnc.updates.rectangles"]
            if info.get("env_status.state_id") is not None:
                to_log["diagnostics/env_state_id"] = info["env_status.state_id"]

        if reward is not None:
            self._episode_reward += reward
            if observation is not None:
                self._episode_length += 1
            self._all_rewards.append(reward)

        if done:
            logger.info('Episode terminating: episode_reward=%s episode_length=%s', self._episode_reward, self._episode_length)
            total_time = time.time() - self._episode_time
            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/episode_length"] = self._episode_length
            to_log["global/episode_time"] = total_time
            to_log["global/reward_per_time"] = self._episode_reward / total_time
            self._episode_reward = 0
            self._episode_length = 0
            self._all_rewards = []

        return observation, reward, done, to_log

def _process_frame42(frame):
    #frame = frame[34:34+160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    #frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 3])
    return frame

class AtariRescale42x42(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [42, 42, 3])

    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]


def create_atari_env(env_id, seed=None):
    env = gym.make(env_id)
    if seed is not None:
        env.seed(seed)
    env = Vectorize(env)
    env = AtariRescale42x42(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    return env

class A3C(object):

    def __init__(self, env, MONITOR_LOCATION, CHECKPOINT_LOCATION, NUM_EPISODES):

        self.MONITOR_LOCATION = MONITOR_LOCATION
        self.chkpt = tf.train.latest_checkpoint(CHECKPOINT_LOCATION)
        self.NUM_EPISODES = NUM_EPISODES
        self.rewards = []
        # logger.info("Loading checkpoint {}".format(chkpt))

    def play(self, num_episodes, env, record=False, seed=None, tst=True):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess, sess.as_default():
            saver = tf.train.import_meta_graph(self.chkpt + ".meta", clear_devices=True)
            g = tf.get_default_graph()
            saver.restore(sess, self.chkpt)
            state_out_0 = tf.get_collection("state_out_0")[0]
            state_out_1 = tf.get_collection("state_out_1")[0]
            get_sample_op = tf.get_collection("greedy_action")[0]
            get_output_state = [state_out_0, state_out_1]

            inp = g.get_tensor_by_name("global/Placeholder:0")
            c_in = g.get_tensor_by_name("global/Placeholder_1:0")
            h_in = g.get_tensor_by_name("global/Placeholder_2:0")

            #
            # print('Trainable vars in {}:'.format(tf.get_variable_scope().name))
            # for v in var_list:
            #     print('  %s %s', v.name, v.get_shape())
            # for tensor in tf.get_default_graph().as_graph_def().node:
            #     print(tensor.name)

            lengths = []
            rewards = []
            self.rewards = []

            if record:
                self.env = wrappers.Monitor(create_atari_env(env, seed),
                                            'result_' + env, force=True)
            else:
                self.env = create_atari_env(env, seed)
            for ep in range(num_episodes):
                obs = self.env.reset()

                initial_state = np.zeros((1,256)).astype(float)
                last_state = [initial_state, initial_state]

                length = 0
                reward_sum = 0
                terminal = False
                while not terminal:
                    feed_dict = {inp : obs[np.newaxis],
                                c_in: last_state[0],
                                h_in: last_state[1]}

                    state, sampled_action = sess.run([get_output_state, get_sample_op], feed_dict = feed_dict)
                    action = sampled_action #.argmax()
                    obs, reward, terminal, info = self.env.step(action)

                    last_state = state
                    length += 1
                    reward_sum += reward
                    self.rewards.append(reward)
                    # print("Episode finished in {} reward: {}".format(length, rewards))
                lengths.append(length)
                print(env + ' UN, score: ', reward_sum)
                rewards.append(reward_sum)
        return sum(rewards)/float(len(rewards))

    def do_submit(self, output, key=''):
        gym.upload(output, api_key=key)


class RandomAgent(object):

    def __init__(self):
        self.env = ''

    def play(self, num_episodes, env, record = False, seed = None):
        self.env = gym.make(env)
        if seed is not None:
            self.env.seed(seed)
        if record:
            self.env = wrappers.Monitor(env, 'result_' + env)
        rewards = []
        for i_episode in range(num_episodes):
            observation = self.env.reset()
            total = 0
            for t in range(10000):
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                total += reward
                if done:
                    break
            rewards.append(total)
            print(env + ' RD, score: ', total)
        return sum(rewards)/float(len(rewards))
    

class Model(ModelDesc):
    def _get_inputs(self):
        assert NUM_ACTIONS is not None
        return [InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputVar(tf.int32, (None,), 'action'),
                InputVar(tf.float32, (None,), 'futurereward')]

    def _get_NN_prediction(self, image):
        image = image / 255.0
        with argscope(Conv2D, nl=tf.nn.relu):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

        l = FullyConnected('fc0', l, 512, nl=tf.identity)
        l = PReLU('prelu', l)
        policy = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)
        return policy

    def _build_graph(self, inputs):
        state, action, futurereward = inputs
        policy = self._get_NN_prediction(state)
        self.logits = tf.nn.softmax(policy, name='logits')


class TPAgent(object):
    def __init__(self, env, MONITOR_LOCATION, CHECKPOINT_LOCATION, NUM_EPISODES):
        self.env = env
        self.NUM_EPISODES = NUM_EPISODES

        self.load = CHECKPOINT_LOCATION
        self.save = MONITOR_LOCATION
        self.predfunc = ''
        self.player = None

        ENV_NAME = self.env
        assert ENV_NAME
        logger.info("Environment Name: {}".format(ENV_NAME))
        p = self.get_player()
        del p    # set NUM_ACTIONS

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'# args.gpu

        self.cfg = PredictConfig(
            model=Model(),
            session_init=SaverRestore(self.load),
            input_names=['state'],
            output_names=['logits'])
        # run_submission(self.cfg, args.output, args.episode)

    def get_player(self, dumpdir=None, seed=None):
        pl = GymEnv(self.env, dumpdir=dumpdir, auto_restart=False, seed=seed)
        pl = MapPlayerState(pl, lambda img: cv2.resize(img, IMAGE_SIZE[::-1]))

        global NUM_ACTIONS
        NUM_ACTIONS = pl.get_action_space().num_actions()

        pl = HistoryFramePlayer(pl, FRAME_HISTORY)
        return pl


    def do_submit(self, output, key=''):
        gym.upload(output, api_key=key)


    def play(self, num_episodes, env, record=False, seed=None, tst=False):
        load = None
        if record:
            load = self.save
        self.player = self.get_player(dumpdir=load, seed=seed)
        if not record:
            self.predfunc = get_predict_func(self.cfg)
        if tst:
            self.predfunc = get_predict_func(self.cfg)
        rewards = []
        if seed is not None:
            print('set seed', seed)
            self.player.player.player.gymenv.seed(seed)
        # logger.info("Start evaluation: ")
        for k in range(num_episodes):
            if k != 0:
                self.player.restart_episode()
            score = play_one_episode(self.player, self.predfunc)
            print(env + ' TP, score: ', score)
            rewards.append(score)
        return sum(rewards)/float(len(rewards))

    