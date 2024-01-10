import argparse

from rllab.algos.tnpg import TNPG
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite

try:
    import seaborn as sns
    sns.set()
except ImportError:
    print('\nConsider installing seaborn (pip install seaborn) for better plotting!')

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("env", help="The environment name from OpenAIGym environments")
parser.add_argument("--n_itr", default=200, type=int)
parser.add_argument("--step_size", default=0.01)
parser.add_argument("--batch_size", default=5000)
parser.add_argument("--gamma", default=.9)
parser.add_argument("--hidden_sizes", default=1, type=int)
parser.add_argument("--data_dir", default="./data_tnpg/")
parser.add_argument("--learn_std", default=True)
parser.add_argument("--init_std", default=1)

args = parser.parse_args()

# ==========================================================================
# OpenAI diabetes envs - HovorkaInterval starts at the same value every time,
# HovorkaIntervalRandom starts at a random value
# ==========================================================================

def run_task(*_):
    env = normalize(GymEnv(args.env))


    baseline = LinearFeatureBaseline(env_spec=env.spec)

    learn_std = args.learn_std
    init_std = args.init_std

    if args.hidden_sizes == 0:
        hidden_sizes=(8,)
    elif args.hidden_sizes == 1:
        hidden_sizes=(32, 32)
    elif args.hidden_sizes == 2:
        hidden_sizes=(100, 50, 25)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=hidden_sizes,
        learn_std=learn_std,
        init_std=init_std
    )

    # =======================
    # Defining the algorithm
    # =======================
    batch_size = args.batch_size
    n_itr = args.n_itr
    gamma = args.gamma
    step_size = args.step_size

    algo = TNPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        n_itr=n_itr,
        discount=gamma,
        step_size=step_size
    )
    algo.train()


# Running and saving the experiment
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
    exp_prefix="TNPG_" + str(args.hidden_sizes),
    # exp_prefix=data_dir
    plot=False
)


