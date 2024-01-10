import argparse

from rllab.algos.ddpg import DDPG
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn (pip install seaborn) for better plotting!')

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("env", help="The environment name from OpenAIGym environments")
parser.add_argument("--reward", default='gaussian', help="The reward function from OpenAIGym diabetes environment")
parser.add_argument("--n_itr", default=200, type=int)
parser.add_argument("--batch_size", default=5000)
parser.add_argument("--gamma", default=.99)
parser.add_argument("--hidden_sizes", default=3, type=int)
parser.add_argument("--data_dir", default="./data_ddpg/")
parser.add_argument("--scale_reward", default=0.1)
parser.add_argument("--init_std", default=1)

args = parser.parse_args()

def run_task(*_):
    env = normalize(GymEnv(args.env, force_reset=True, record_video=False))
    env.wrapped_env.env.env.reward_flag = args.reward

    if args.hidden_sizes == 0:
        hidden_sizes=(8,)
    elif args.hidden_sizes == 1:
        hidden_sizes=(32, 32)
    elif args.hidden_sizes == 2:
        hidden_sizes=(100, 50, 25)
    elif args.hidden_sizes == 3:
        hidden_sizes=(400, 300)

    policy = DeterministicMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=hidden_sizes
    )

    es = OUStrategy(env_spec=env.spec)

    qf = ContinuousMLPQFunction(env_spec=env.spec)

    algo = DDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        batch_size=64,
        max_path_length=95,
        epoch_length=args.batch_size,
        min_pool_size=10000,
        n_epochs=args.n_itr,
        discount=args.gamma,
        scale_reward=args.scale_reward,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4,
        eval_samples=95,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()

run_experiment_lite(
    run_task,
    # algo.train(),
    log_dir=args.data_dir,
    # n_parallel=2,
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    exp_prefix="DDPG" + str(args.hidden_sizes),
    # exp_prefix=data_dir
    plot=False
)

