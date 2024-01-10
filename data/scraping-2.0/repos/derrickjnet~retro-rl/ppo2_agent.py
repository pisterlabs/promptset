#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""

import os
import tensorflow as tf

import ppo2.ppo2 as ppo2
#import baselines.ppo2.ppo2 as ppo2
import ppo2.policies as policies
#import baselines.ppo2.policies as policies
#import baselines.logger as logger
import gym_remote.exceptions as gre
from baselines.common.atari_wrappers import FrameStack

from vec_env.reward_scaling_vec_env import RewardScalingVecEnv
from exploration.exploration_vec_env import ExplorationVecEnv
from exploration.exploration import Exploration
from exploration.state_encoder import StateEncoder
from sonic_util import make_vec_env

def main():
    discount = os.environ.get('RETRO_DISCOUNT')
    if discount != None:
      discount = float(discount)
    else:
      discount=0.99
    print("DISCOUNT: %s" % (discount,))

    vec_env = make_vec_env(extra_wrap_fn=lambda env: FrameStack(env, 4))

    """Run PPO until the environment throws an exception."""
    #logger.configure(dir=os.environ.get('RETRO_CHECKPOINT_DIR'))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        if 'RETRO_ENCODER_DIR' in os.environ:
          state_encoder = StateEncoder(sess, encoder_dir = os.environ['RETRO_ENCODER_DIR'])
        else:
          state_encoder = None

        def init_fun():
          if state_encoder != None:
            state_encoder.initialize()
          if "RETRO_INIT_DIR" in os.environ:
            saver = tf.train.Saver(var_list=tf.trainable_variables('ppo2_model'))
            latest_checkpoint = tf.train.latest_checkpoint(os.environ['RETRO_INIT_DIR'])
            print("PPO2_LOAD_INIT_CHECKPOINT: %s" % (latest_checkpoint,))
            saver.restore(sess, latest_checkpoint)
            #from tensorflow.python.tools import inspect_checkpoint as chkp
            #chkp.print_tensors_in_checkpoint_file(latest_checkpoint,'',all_tensors=True)

        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2.learn(policy=policies.CnnPolicy,
                   env=RewardScalingVecEnv(ExplorationVecEnv(vec_env, Exploration, state_encoder=state_encoder), reward_scale = 0.01),
                   nsteps=4096,
                   nminibatches=8,
                   lam=0.95,
                   gamma=discount, #0.99
                   noptepochs=3,
                   log_interval=1,
                   ent_coef=0.01,
                   lr=lambda _: 2e-4,
                   cliprange=lambda _: 0.1,
                   total_timesteps=int(1.5e6 * vec_env.num_envs),
                   save_interval=1,
                   init_fun=init_fun)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
