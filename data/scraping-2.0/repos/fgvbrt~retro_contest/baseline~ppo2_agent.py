#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""

import tensorflow as tf

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import baselines.ppo2.ppo2 as ppo2
import gym_remote.exceptions as gre
import functools
import argparse
import sonic_util
from baselines import logger
from baselines.ppo2.policies import LstmPolicy, CnnPolicy
import utils
import os
import yaml
import warnings
from datetime import datetime


def add_boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.
    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def main(policy, env, params):
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2.learn(policy=policy,
                   env=env,
                   nsteps=params['n_steps'],
                   nminibatches=(params['n_steps']*env.num_envs) // params["batch_size"],
                   lam=params["lam"],
                   gamma=params['gamma'],
                   noptepochs=params["n_opt_epochs"],
                   log_interval=params["log_interval"],
                   ent_coef=params["ent_coef"],
                   vf_coef=params['vf_coef'],
                   lr=lambda _: params["lr"],
                   cliprange=lambda _: params['cliprange'],
                   max_grad_norm=params['max_grad_norm'],
                   total_timesteps=params["max_steps"],
                   save_interval=params["save_interval"],
                   weights_path=params["weights_path"],
                   adam_stats=params["adam_stats"],
                   nmixup=params["nmixup"],
                   weights_choose_eps=params["weights_choose_eps"],
                   cnn=params['cnn'])


def run_train():
    def _parse_args():
        parser = argparse.ArgumentParser(description="Run commands")
        parser.add_argument('--config', type=str, default=None, nargs='+',
                            help="file with config")
        return parser.parse_args()

    args = _parse_args()
    config = utils.load_config(args.config)

    env_params = config['env_params']
    train_params = config['train_params']

    if train_params["policy"] == 'lstm':
        policy = LstmPolicy
    elif train_params["policy"] == 'cnn':
        policy = CnnPolicy
    else:
        raise ValueError("unknown policy {}".format(train_params["policy"]))

    if train_params['cnn'] == "openai1" and not env_params['small_size']:
        warnings.warn('asked for openai1 policy, but dont set small size for env params')

    # create environments funcitons
    n_envs = train_params['n_envs']
    if n_envs == 1:
        vec_fn = DummyVecEnv
    elif n_envs > 1:
        vec_fn = SubprocVecEnv
    else:
        raise ValueError('number of environments less than 1: {}'.format(n_envs))
    env = vec_fn([functools.partial(sonic_util.make_from_config, env_params) for _ in range(n_envs)])

    logdir = os.path.join("logs", str(datetime.now()))
    logger.configure(logdir)

    # save run config
    with open(os.path.join(logdir, "run_config.yaml"), 'w') as f:
        yaml.dump(config, f)

    main(policy, env, train_params)


if __name__ == '__main__':
    try:
        run_train()
    except gre.GymRemoteError as exc:
        print('exception', exc)
