import argparse
import os.path as osp

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite, stub
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc import ext

# Relu nonlinearity
import lasagne.nonlinearities as NL

# from pylab import plot, figure, show, title

parser = argparse.ArgumentParser()
parser.add_argument("env", help="The environment name from OpenAIGym environments")
parser.add_argument("--reward", default='absolute', help="The reward function from OpenAIGym diabetes environment")
parser.add_argument("--num_epochs", default=250, type=int)
parser.add_argument("--data_dir", default="./data_trpo/")

args = parser.parse_args()

stub(globals())
ext.set_seed(1)

# gymenv = GymEnv(args.env)
gymenv = GymEnv(args.env, force_reset=True, record_video=False, record_log=True)

env = (normalize(gymenv))

env.wrapped_env.env.env.reward_flag = args.reward

policy = GaussianMLPPolicy(
env_spec=env.spec,
# The neural network policy should have two hidden layers, each with 32 hidden units.
hidden_sizes=(100, 50, 25),
hidden_nonlinearity=NL.rectify,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=5000,
    max_path_length=env.horizon,
    n_itr=args.num_epochs,
    discount=0.99,
    step_size=0.01,
)

run_experiment_lite(
    algo.train(),
    log_dir=args.data_dir,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    exp_prefix="TRPO_" + args.env,
    seed=1,
    mode="local",
    plot=False,
    # terminate_machine=args.dont_terminate_machine,
    added_project_directories=[osp.abspath(osp.join(osp.dirname(__file__), '.'))]
)
