# Hardcoded script to run experiment/analysis
# of the quality of training UBMs/doing MAP-adaptations
# with different amount of data.

import os
from glob import glob
from multiprocessing import pool

import numpy as np
from tqdm import tqdm

from gmm_tools import trajectories_to_supervector, load_ubm, adapted_gmm_distance


# Different number of trajectories
# per policy used to train UBMs/MAPs.
NUM_TRAJECTORIES = [
    100,
    50,
    25,
    10
]

# Different number of components used
# to train UBMs
NUM_COMPONENTS = [
    64,
    32,
    16,
    8,
    4,
    2,
    1
]

# Different envs to analyze over
ENVS = [
    "Pendulum-v0",
    "CartPole-v1",
    "Acrobot-v1",
    "BipedalWalker-v3",
    "LunarLander-v2"
]

# Reward scales (min, max) for each
# environment.
# Min is looking at what policies are initially at,
# max is from openai gym "reward threshold" for solving the env:
#  https://github.com/openai/gym/wiki/Table-of-environments
REWARD_SCALES = {
    "Pendulum-v0": [-1600, -200],
    "Acrobot-v1": [-500, -100],
    "LunarLander-v2": [-230, 200],
    "BipedalWalker-v3": [-100, 300],
    "CartPole-v1": [0, 500]
}

# If two policy's average return, normalized
# by above min/max, differs more than this, then
# they are considered different.
DIFFERENT_REWARD_THRESHOLD = 0.25


# Number of different policies analyzed
NUM_POLICIES_ANALYZED = 3

# Number of times each experiment is
# repeated
NUM_REPETITIONS = 3

# Number of checkpoints used per experiment for the
# comparisons
NUM_CHECKPOINTS = 100

# Template for loading a specific UBM
UBM_TEMPLATE = "ubm_experiments/ubms_{num_components}_components_{num_traj}_trajectories/experiments/stablebaselines_{env}_{policy_name}_repetition_{repetition_num}/{env}_ubm.npz"
# Template to list of directories of specific experiment
TRAJECTORY_TEMPLATE = "experiments/stablebaselines_{env}_{policy_name}/trajectories/"
# Directory where to output results for caching
OUTPUT_DIRECTORY = "ubm_experiments/analyzing_cache"
PIVECTOR_TEMPLATE = os.path.join(OUTPUT_DIRECTORY, "{env}_{num_components}_{num_traj}_{policy_name}_{repetition_num}_pivectors.npz")
DISTANCE_MATRIX_TEMPLATE = os.path.join(OUTPUT_DIRECTORY, "{env}_{num_components}_{num_traj}_{policy_name}_{repetition_num}_distances.npz")
# Directory where to put figures
FIGURE_DIRECTORY = "figures"


def extract_pivector_worker(num_traj_index, num_traj, num_components, env):
    # Worker for the function below
    trained_ubms = glob(UBM_TEMPLATE.format(num_traj=num_traj, num_components=num_components, env=env, policy_name="*", repetition_num="*"))
    trained_ubm_dirs = [os.path.basename(os.path.dirname(x)) for x in trained_ubms]
    policy_names = ["_".join(x.split("_")[-4:-2]) for x in trained_ubm_dirs]
    policy_names = sorted(list(set(policy_names)))
    for policy_name in policy_names:
        for repetition in range(1, NUM_REPETITIONS + 1):
            pivector_path = PIVECTOR_TEMPLATE.format(num_traj=num_traj, num_components=num_components, env=env, policy_name=policy_name, repetition_num=repetition)
            # If already exists, skip extracting pivectors for this
            if os.path.isfile(pivector_path):
                continue
            # Load UBM
            ubm_path = UBM_TEMPLATE.format(num_traj=num_traj, num_components=num_components, env=env, policy_name=policy_name, repetition_num=repetition)
            ubm, means, stds = load_ubm(ubm_path)
            # Hacky thing to load the same trajectories as used in UBM training
            ubm_data = np.load(ubm_path)
            trajectory_indeces = None
            if "trajectory_indeces" in ubm_data.keys():
                trajectory_indeces = ubm_data["trajectory_indeces"]
            else:
                # A wupsie...
                trajectory_indeces = ubm_data["random_trajectory_indeces"]
            ubm_data.close()
            # Load trajectory data
            trajectories_path = glob(os.path.join(TRAJECTORY_TEMPLATE.format(env=env, policy_name=policy_name), "*"))
            trajectories_path = sorted(trajectories_path)
            all_pivectors = []
            all_average_episodic_returns = []
            all_trajectory_data = []
            for trajectory_i, trajectory_path in enumerate(trajectories_path):
                data = np.load(trajectory_path)
                keys = sorted(list(data.keys()))
                all_average_episodic_returns.append(data["episodic_rewards"].mean())
                # Take trajectories at same indeces as in used in training UBM.
                # First make sure it is in same order as with ubm training
                datas = [data[key] for key in keys if "traj" in key]
                datas = [datas[i] for i in trajectory_indeces[trajectory_i]]

                data = np.concatenate(datas, axis=0)
                data = (data - means) / stds
                all_trajectory_data.append(data)
                pivector = trajectories_to_supervector(data, ubm)
                all_pivectors.append(pivector)
            all_pivectors = np.array(all_pivectors)

            np.savez(
                pivector_path,
                pivectors=all_pivectors,
                average_episodic_rewards=all_average_episodic_returns,
                covariances=ubm.covariances_,
                weights=ubm.weights_,
            )


def extract_pivectors_worker_sugarcoat(kwargs):
    # A simple kwargs unpacker for extract_pivectors_worker
    extract_pivector_worker(**kwargs)


def extract_pivectors():
    # Extract pi-vectors for different amount of trajectories etc.

    jobs = []
    for num_traj_index, num_traj in enumerate(NUM_TRAJECTORIES):
        for num_components in NUM_COMPONENTS:
            for env in ENVS:
                jobs.append(dict(
                    num_traj_index=num_traj_index,
                    num_traj=num_traj,
                    num_components=num_components,
                    env=env
                ))
    workers = pool.Pool()
    for _ in tqdm(workers.imap(extract_pivectors_worker_sugarcoat, jobs), total=len(jobs)):
        pass
    workers.close()


def compute_distance_matrices():
    # Compute distance matrices for each pivector file
    pivector_files = glob(PIVECTOR_TEMPLATE.format(env="*", num_traj="*", num_components="*", policy_name="*", repetition_num="*"))
    for pivector_file in tqdm(pivector_files, desc="distance"):
        # Skip if exists
        # Distance file name is same as pivectors, but replace "pivector" with "distance"
        distance_file = pivector_file.replace("pivectors", "distances")
        if os.path.isfile(distance_file):
            continue

        data = np.load(pivector_file)
        pivectors = data["pivectors"]
        covariances = data["covariances"]
        precisions = 1 / covariances
        weights = data["weights"]
        num_pivectors = len(pivectors)
        mean_shape = precisions.shape

        # Create with np.ones to allocate space, so
        # know immediattely if we are running out of space.
        distance_matrix = np.ones((num_pivectors, num_pivectors))
        for i in range(num_pivectors):
            # Cut ~half of the computation needed
            for j in range(i, num_pivectors):
                means1 = pivectors[i].reshape(mean_shape)
                means2 = pivectors[j].reshape(mean_shape)
                distance = adapted_gmm_distance(means1, means2, precisions, weights)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        np.savez(
            distance_file,
            distance_matrix=distance_matrix,
            average_episodic_rewards=data["average_episodic_rewards"]
        )


def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    # Extract pivectors for each repetition+traj_num+component+policy+env
    # combination.
    extract_pivectors()
    # Compute distance matrices for each pivector file.
    # Each file contains multiple pivectors and we
    # want to see if distances between them change between
    # runs
    compute_distance_matrices()


if __name__ == "__main__":
    main()
