#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""

import tensorflow as tf

# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
import gym_remote.exceptions as gre
import env_servers
import functools
import argparse
import sonic_util
import pandas as pd
from time import sleep
from baselines import logger
from multiprocessing import cpu_count


def main(clients_fn):
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2.learn(policy=policies.CnnPolicy,
                   env=SubprocVecEnv(clients_fn),
                   nsteps=4096,
                   nminibatches=8,
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=3,
                   log_interval=1,
                   ent_coef=0.01,
                   lr=lambda _: 2e-4,
                   cliprange=lambda _: 0.1,
                   total_timesteps=int(10e7),
                   save_interval=10)


def run_dummy():
    def _parse_args():
        parser = argparse.ArgumentParser(description="Run commands")
        parser.add_argument('--socket_dir', type=str, default='/tmp/retro', help="Base directory for sockers.")
        parser.add_argument('--csv_file', type=str, default='all.csv', help="Csv file with train games.")
        parser.add_argument('--steps', type=int, default=None, help="Number of timestemps for each environment.")
        return parser.parse_args()

    args = _parse_args()
    game_states = pd.read_csv(args.csv_file).values

    env_process, game_dirs = env_servers.start_servers(game_states, args.socket_dir, args.steps)

    clients_fn = [functools.partial(sonic_util.make_remote_env, socket_dir=d) for d in game_dirs]

    sleep(2)
    logger.configure('logs')
    main(clients_fn)

    for p in env_process:
        p.join()


def run_subprocess():
    def _parse_args():
        parser = argparse.ArgumentParser(description="Run commands")
        parser.add_argument('--csv_file', type=str, default='all.csv', help="Csv file with train games.")
        parser.add_argument('--num_envs', type=int, default=int(1.5*cpu_count()) - 1, help="Number of parallele environments.")
        return parser.parse_args()

    args = _parse_args()
    game_states = pd.read_csv(args.csv_file).values.tolist()

    clients_fn = [functools.partial(sonic_util.make_rand_env, game_states) for _ in range(args.num_envs)]

    sleep(2)
    logger.configure('logs')
    main(clients_fn)


if __name__ == '__main__':
    try:
        run_subprocess()
    except gre.GymRemoteError as exc:
        print('exception', exc)
