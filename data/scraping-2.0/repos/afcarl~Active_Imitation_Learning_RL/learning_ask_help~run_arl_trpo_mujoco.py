from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.envs.base import TfEnv
# Policies
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from shared_gaussian_mlp_policy import LayeredGaussianMLPPolicy
### TRPO Classes
from sandbox.rocky.tf.algos.trpo import TRPO as Oracle_TRPO
from agent_trpo import TRPO
from rllab.misc import ext
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
import rllab.misc.logger as logger
import pickle
import os.path as osp
import numpy as np
from all_utilities import *
import tensorflow as tf
import argparse
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.box import Box

parser = argparse.ArgumentParser()
parser.add_argument("envs", help="The environment name from OpenAIGym environments")
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--text_log_file", default="./data/debug.log", help="Where text output will go")
parser.add_argument("--tabular_log_file", default="./data/progress.csv", help="Where tabular output will go")
parser.add_argument("--text_log_file_active", default="./data/debug.log", help="Where text output will go")
parser.add_argument("--tabular_log_file_active", default="./data/progress_active.csv", help="Where tabular output will go")
args = parser.parse_args()

logger.add_text_output(args.text_log_file)
logger.add_tabular_output(args.tabular_log_file)
logger.set_log_tabular_only(False)


supported_gym_envs = ["MountainCar-v0", "InvertedPendulum-v1", "InvertedDoublePendulum-v1", "Hopper-v1", "HalfCheetah-v1"]

gymenv = GymEnv(args.envs, force_reset=True, record_video=False, record_log=False)
env = TfEnv(gymenv)


"""
ORACLE POLICY
"""
oracle_policy = GaussianMLPPolicy(
    name="oracle_policy",
    env_spec=env.spec,
    hidden_sizes=(100, 50, 25),
    hidden_nonlinearity=tf.nn.relu,
)

oracle_baseline = LinearFeatureBaseline(env_spec=env.spec)

"""
AGENT POLICY
"""
policy = LayeredGaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(100, 50, 25),
    hidden_nonlinearity=tf.nn.relu,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)


"""
Few hyperparameters
"""
max_path_length_horizon = 2000
num_final_rollouts = 10
batch_size_value = 5000
step_size_value = 0.01
regularisation_coefficient = 1e-5


with tf.Session() as sess:
    oracle_algo = Oracle_TRPO(
        env=env,
        policy=oracle_policy,
        baseline=oracle_baseline,
        batch_size=batch_size_value,    #use batch size upto 25000
        max_path_length=max_path_length_horizon, #or use env.horizon here - would be suited for different environments (may not be defined for all envs though)
        n_itr=args.num_epochs,
        discount=0.99,
        step_size=step_size_value,
        optimizer=ConjugateGradientOptimizer(reg_coeff=regularisation_coefficient, hvp_approach=FiniteDifferenceHvp(base_eps=regularisation_coefficient))
    )

    oracle_train(oracle_algo, sess=sess)


    # rollouts = oracle_algo.obtain_samples(num_epochs + 1)
    #logger.log("Average reward for training rollouts on (%s): %f +- %f " % (env_name, np.mean([np.sum(p['rewards']) for p in rollouts]),  np.std([np.sum(p['rewards']) for p in rollouts])))

    """
    Evaluating the learnt policy below
    using the "obtaines_samples" collected from above

    batch_polopt.py
    """
    # Final evaluation on all environments using the learned policy
    # total_rollouts = []
    # # for env_name, env in envs:
    # rollouts = []
    # for i in range(num_final_rollouts):
    #     rollout = rollout_policy(oracle_policy, env, max_path_length=max_path_length_horizon, speedup=1, get_image_observations=False, animated=False)
    #     rollouts.append(rollout)
    #     total_rollouts.append(rollout)

    # logger.log("Average reward for eval rollouts on (%s): %f +- %f " % (env_name, np.mean([np.sum(p['rewards']) for p in rollouts]),  np.std([np.sum(p['rewards']) for p in rollouts])))

    # logger.log("Total Average across all rollouts and envs: %f +- %f " % (np.mean([np.sum(p['rewards']) for p in total_rollouts]),  np.std([np.sum(p['rewards']) for p in total_rollouts])))


    print ("Oracle TRPO Policy has been trained")

    logger.add_text_output(args.text_log_file_active)
    logger.add_tabular_output(args.tabular_log_file_active)
    logger.set_log_tabular_only(False)


    """
    Use the learnt policy (Oracle Policy) for TRPO-Active-Learning
    """
    algo = TRPO(
        sess=sess,
        env=env,
        policy=policy,
        oracle_policy=oracle_policy,
        baseline=baseline,
        batch_size=batch_size_value,
        max_path_length = max_path_length_horizon,         #max_path_length=env.horizon,
        n_itr=args.num_epochs,
        discount=0.99,
        step_size=step_size_value,
        gae_lambda=1.0,
        optimizer=ConjugateGradientOptimizer(reg_coeff=regularisation_coefficient, hvp_approach=FiniteDifferenceHvp(base_eps=regularisation_coefficient))
    ) 
    agent_train(algo, oracle_policy=oracle_policy, sess=sess )    

    # """
    # Furher need to evaluate the learnt agent policy
    # - taken from batch_polopt_active.py
    # """

    # total_rollouts = []
    # rollouts = []

    # for i in range(num_final_rollouts):
    #     rollout = rollout_policy(policy, env_modified, max_path_length=max_path_length_horizon, speedup=1, get_image_observations=False, animated=False)
    #     rollouts.append(rollout)
    #     total_rollouts.append(rollout)

    # logger.log("Average reward for eval rollouts on (%s): %f +- %f " % (env_name_modified, np.mean([np.sum(p['rewards']) for p in rollouts]),  np.std([np.sum(p['rewards']) for p in rollouts])))
    # logger.log("Total Average across all rollouts and envs: %f +- %f " % (np.mean([np.sum(p['rewards']) for p in total_rollouts]),  np.std([np.sum(p['rewards']) for p in total_rollouts])))





