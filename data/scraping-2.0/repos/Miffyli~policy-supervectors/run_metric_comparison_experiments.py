# Hardcoded script to run
# comparisons between different behavioural embeddings
# using different metrics.
# Note we use UBM data directory again here (bit of overloading).
import os
from glob import glob
from multiprocessing import pool

import numpy as np
from tqdm import tqdm
import torch as th

from gmm_tools import trajectories_to_supervector, load_ubm, adapted_gmm_distance


# Different number of trajectories
# per policy used to train things.
NUM_TRAJECTORIES = [
    100,
    50,
    25,
    10
]

# Number of bins to use per axis for discretization run
NUM_DISCRETE_BINS = 10

# Use a fixed number of components for our policy-supervector's results.
# Why single number?
#   We want to use same trajectory data for all methods, which is contained
#   in the UBM training file. We need to know exactly which UBM file we need
#   for loading trajectories for other methods
NUM_COMPONENTS = 64

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

# Number of different policies analyzed
NUM_POLICIES_ANALYZED = 3

# Number of times each experiment is
# repeated
NUM_REPETITIONS = 3

# Number of checkpoints used per experiment for the
# comparisons
NUM_CHECKPOINTS = 100

# Epsilon for covariance matrices
EPS = 1e-7

# Template for loading a specific UBM (for policy-supervectors)
UBM_TEMPLATE = "ubm_experiments/ubms_{num_components}_components_{num_traj}_trajectories/experiments/stablebaselines_{env}_{policy_name}_repetition_{repetition_num}/{env}_ubm.npz"
# Template to list of directories of specific experiment
TRAJECTORY_TEMPLATE = "experiments/stablebaselines_{env}_{policy_name}/trajectories/"
# Directory where to output results for caching
OUTPUT_DIRECTORY = "ubm_experiments/analyzing_cache"
PIVECTOR_TEMPLATE = os.path.join(OUTPUT_DIRECTORY, "{env}_{num_components}_{num_traj}_{policy_name}_{repetition_num}_pivectors.npz")
GAUSSIAN_TEMPLATE = os.path.join(OUTPUT_DIRECTORY, "{env}_{num_traj}_{policy_name}_{repetition_num}_gaussians.npz")
PIVECTOR_DISTANCE_MATRIX_TEMPLATE = os.path.join(OUTPUT_DIRECTORY, "{env}_{num_components}_{num_traj}_{policy_name}_{repetition_num}_pivector_distances.npz")
GAUSSIAN_DISTANCE_MATRIX_TEMPLATE = os.path.join(OUTPUT_DIRECTORY, "{env}_{num_traj}_{policy_name}_{repetition_num}_gaussian_distances.npz")
DISCRIMINATOR_DISTANCE_MATRIX_TEMPLATE = os.path.join(OUTPUT_DIRECTORY, "{env}_{num_traj}_{policy_name}_{repetition_num}_discriminator_distances.npz")
ENCODER_DISTANCE_MATRIX_TEMPLATE = os.path.join(OUTPUT_DIRECTORY, "{env}_{num_traj}_{policy_name}_{repetition_num}_encoder_distances.npz")
DISCRETIZATION_DISTANCE_MATRIX_TEMPLATE = os.path.join(OUTPUT_DIRECTORY, "{env}_{num_traj}_{policy_name}_{repetition_num}_discretization_distances.npz")
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
            trajectory_indeces = ubm_data["trajectory_indeces"]
            ubm_data.close()
            # Load trajectory data
            trajectories_path = glob(os.path.join(TRAJECTORY_TEMPLATE.format(env=env, policy_name=policy_name), "*"))
            trajectories_path = sorted(trajectories_path)
            all_pivectors = []
            all_average_episodic_returns = []
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
        for env in ENVS:
            jobs.append(dict(
                num_traj_index=num_traj_index,
                num_traj=num_traj,
                num_components=NUM_COMPONENTS,
                env=env
            ))
    workers = pool.Pool()
    for _ in tqdm(workers.imap(extract_pivectors_worker_sugarcoat, jobs), total=len(jobs), desc="pivectors"):
        pass
    workers.close()


def extract_gaussian_worker(num_traj_index, num_traj, env):
    # First get list of different policy names we have so we can iterate over them
    # (We are not actually using UBMs  here)
    trained_ubms = glob(UBM_TEMPLATE.format(num_traj=num_traj, num_components=NUM_COMPONENTS, env=env, policy_name="*", repetition_num="*"))
    trained_ubm_dirs = [os.path.basename(os.path.dirname(x)) for x in trained_ubms]
    policy_names = ["_".join(x.split("_")[-4:-2]) for x in trained_ubm_dirs]
    policy_names = sorted(list(set(policy_names)))
    assert len(policy_names) == NUM_POLICIES_ANALYZED

    for policy_name in policy_names:
        for repetition in range(1, NUM_REPETITIONS + 1):
            gaussian_path = GAUSSIAN_TEMPLATE.format(num_traj=num_traj, num_components=NUM_COMPONENTS, env=env, policy_name=policy_name, repetition_num=repetition)
            # If already exists, skip extracting pivectors for this
            if os.path.isfile(gaussian_path):
                continue

            # Hacky thing to load up which trajectories were sampled.
            ubm_path = UBM_TEMPLATE.format(num_traj=num_traj, num_components=NUM_COMPONENTS, env=env, policy_name=policy_name, repetition_num=repetition)
            ubm_data = np.load(ubm_path)
            trajectory_indeces = ubm_data["trajectory_indeces"]
            ubm_data.close()

            # Load trajectory data
            trajectories_path = glob(os.path.join(TRAJECTORY_TEMPLATE.format(env=env, policy_name=policy_name), "*"))
            trajectories_path = sorted(trajectories_path)
            all_means = []
            all_stds = []
            all_average_episodic_returns = []
            for trajectory_i, trajectory_path in enumerate(trajectories_path):
                data = np.load(trajectory_path)
                keys = sorted(list(data.keys()))
                all_average_episodic_returns.append(data["episodic_rewards"].mean())
                # Take trajectories at same indeces as in used in training UBM.
                # First make sure it is in same order as with ubm training
                datas = [data[key] for key in keys if "traj" in key]
                datas = [datas[i] for i in trajectory_indeces[trajectory_i]]

                data = np.concatenate(datas, axis=0)
                all_means.append(np.mean(data, axis=0))
                all_stds.append(np.std(data, axis=0))
            all_means = np.array(all_means)
            all_stds = np.array(all_stds)

            np.savez(
                gaussian_path,
                means=all_means,
                stds=all_stds,
                average_episodic_rewards=all_average_episodic_returns,
            )


def extract_gaussians_worker_sugarcoat(kwargs):
    # A simple kwargs unpacker for extract_gaussians_worker
    extract_gaussian_worker(**kwargs)


def extract_gaussians():
    # Extract single gaussians
    jobs = []
    for num_traj_index, num_traj in enumerate(NUM_TRAJECTORIES):
        for env in ENVS:
            jobs.append(dict(
                num_traj_index=num_traj_index,
                num_traj=num_traj,
                env=env
            ))
    workers = pool.Pool()
    for _ in tqdm(workers.imap(extract_gaussians_worker_sugarcoat, jobs), total=len(jobs), desc="gaussians"):
        pass
    workers.close()


def compute_pivector_distance_matrices():
    # Compute distance matrices for each pivector file
    pivector_files = glob(PIVECTOR_TEMPLATE.format(env="*", num_traj="*", num_components="*", policy_name="*", repetition_num="*"))
    for pivector_file in tqdm(pivector_files, desc="distance"):
        # Skip if exists
        # Distance file name is same as pivectors, but replace "pivector" with "distance"
        distance_file = pivector_file.replace("pivectors", "pivector_distances")
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


def compute_gaussian_distance_matrices():
    # Compute distance matrices for each gaussian file
    gaussian_files = glob(GAUSSIAN_TEMPLATE.format(env="*", num_traj="*", policy_name="*", repetition_num="*"))
    for gaussian_file in tqdm(gaussian_files, desc="distance"):
        # Skip if exists
        # Distance file name is same as pivectors, but replace "pivector" with "distance"
        distance_file = gaussian_file.replace("gaussians", "gaussian_distances")
        if os.path.isfile(distance_file):
            continue

        data = np.load(gaussian_file)

        means = data["means"]
        stds = data["stds"]
        num_pivectors = len(means)

        # Create with np.ones to allocate space, so
        # know immediattely if we are running out of space.
        distance_matrix = np.ones((num_pivectors, num_pivectors))
        for i in range(num_pivectors):
            # Cut ~half of the computation needed
            dist_i = th.distributions.MultivariateNormal(
                th.as_tensor(means[i]),
                th.as_tensor(np.diag(stds[i] ** 2 + EPS))
            )
            for j in range(i, num_pivectors):
                dist_j = th.distributions.MultivariateNormal(
                    th.as_tensor(means[j]),
                    th.as_tensor(np.diag(stds[j] ** 2 + EPS))
                )

                # Standard KL-divergence between two gaussians.
                # Using Torch implementation (I trust such code base).
                # Compute both ways and sum together for symmetric
                # value.
                distance = th.distributions.kl_divergence(dist_i, dist_j) + th.distributions.kl_divergence(dist_j, dist_i)

                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        np.savez(
            distance_file,
            distance_matrix=distance_matrix,
            average_episodic_rewards=data["average_episodic_rewards"]
        )


def compute_discriminator_distance_worker(num_traj_index, num_traj, env):
    # Import a library we need here
    from discriminator_tools import train_discriminator
    th.set_num_threads(1)

    # First get list of different policy names we have so we can iterate over them
    # (We are not actually using UBMs  here)
    trained_ubms = glob(UBM_TEMPLATE.format(num_traj=num_traj, num_components=NUM_COMPONENTS, env=env, policy_name="*", repetition_num="*"))
    trained_ubm_dirs = [os.path.basename(os.path.dirname(x)) for x in trained_ubms]
    policy_names = ["_".join(x.split("_")[-4:-2]) for x in trained_ubm_dirs]
    policy_names = sorted(list(set(policy_names)))
    assert len(policy_names) == NUM_POLICIES_ANALYZED

    for policy_name in policy_names:
        for repetition in range(1, NUM_REPETITIONS + 1):
            discriminator_distance_path = DISCRIMINATOR_DISTANCE_MATRIX_TEMPLATE.format(num_traj=num_traj, env=env, policy_name=policy_name, repetition_num=repetition)
            # If already exists, skip extracting pivectors for this
            if os.path.isfile(discriminator_distance_path):
                continue

            # Hacky thing to load up which trajectories were sampled.
            ubm_path = UBM_TEMPLATE.format(num_traj=num_traj, num_components=NUM_COMPONENTS, env=env, policy_name=policy_name, repetition_num=repetition)
            ubm_data = np.load(ubm_path)
            trajectory_indeces = ubm_data["trajectory_indeces"]
            ubm_data.close()

            # Load trajectory data
            trajectories_path = glob(os.path.join(TRAJECTORY_TEMPLATE.format(env=env, policy_name=policy_name), "*"))
            trajectories_path = sorted(trajectories_path)
            policy_datas = []
            all_average_episodic_returns = []
            for trajectory_i, trajectory_path in enumerate(trajectories_path):
                data = np.load(trajectory_path)
                keys = sorted(list(data.keys()))
                all_average_episodic_returns.append(data["episodic_rewards"].mean())
                # Take trajectories at same indeces as in used in training UBM.
                # First make sure it is in same order as with ubm training
                datas = [data[key] for key in keys if "traj" in key]
                datas = [datas[i] for i in trajectory_indeces[trajectory_i]]
                data = np.concatenate(datas, axis=0).astype(np.float32)
                policy_datas.append(data)

            # Now the fun part: For each policy,
            #   train network to separate that policy from the remaining
            #   data, then test for each policy the density-ratio of
            #   states agent visits

            num_pivectors = len(policy_datas)

            distance_matrix = np.zeros((num_pivectors, num_pivectors))
            for i in range(num_pivectors):
                target_data = policy_datas[i]
                non_target_data = np.concatenate([policy_datas[non_i] for non_i in range(num_pivectors) if non_i != i], axis=0)
                discriminator = train_discriminator(target_data, non_target_data)
                # Go over the full matrix as we need
                # to add distances bot ways.
                for j in range(num_pivectors):
                    # See what is the probability ratio
                    # p_target(s) / p_{non_target}(s) (= d(s) / (1 - d(s) ))
                    # for the data from other policy.
                    # If above is high -> close to original data (likely same policy),
                    # if close to zero -> far from original data
                    policy_j_data = policy_datas[j]
                    d_policy_j_data = None
                    with th.no_grad():
                        d_policy_j_data = discriminator(th.as_tensor(policy_j_data).float()).detach().numpy()
                    # Discriminator has a clamp on the values so d_policy_j_data can not be zero,
                    # so no need to add eps here
                    density_ratio = d_policy_j_data / (1 - d_policy_j_data)

                    # Mapping to a distance value (higher value -> further away)
                    distances = np.exp(-np.log(density_ratio))

                    distance = np.mean(distances)

                    # We add distances values from both ways
                    # around (similar to KL-distance) for
                    # symmetric matrix.
                    distance_matrix[i, j] += distance
                    distance_matrix[j, i] += distance
            np.savez(
                discriminator_distance_path,
                distance_matrix=distance_matrix,
                average_episodic_rewards=all_average_episodic_returns
            )


def compute_discriminator_distances_worker_sugarcoat(kwargs):
    # A simple kwargs unpacker for compute_discriminator_distances_worker
    compute_discriminator_distance_worker(**kwargs)


def compute_discriminator_distance_matrices():
    # We need to do heavy work here so parallelize in the
    # dirty way we did before
    jobs = []
    for num_traj_index, num_traj in enumerate(NUM_TRAJECTORIES):
        for env in ENVS:
            jobs.append(dict(
                num_traj_index=num_traj_index,
                num_traj=num_traj,
                env=env
            ))
    workers = pool.Pool()
    for _ in tqdm(workers.imap(compute_discriminator_distances_worker_sugarcoat, jobs), total=len(jobs), desc="discriminators"):
        pass
    workers.close()


def compute_encoder_distance_worker(num_traj_index, num_traj, env):
    # Import a library we need here
    from trajectory_latent_tools import train_trajectory_encoder, encode_policy_into_gaussian
    th.set_num_threads(2)

    # First get list of different policy names we have so we can iterate over them
    # (We are not actually using UBMs  here)
    trained_ubms = glob(UBM_TEMPLATE.format(num_traj=num_traj, num_components=NUM_COMPONENTS, env=env, policy_name="*", repetition_num="*"))
    trained_ubm_dirs = [os.path.basename(os.path.dirname(x)) for x in trained_ubms]
    policy_names = ["_".join(x.split("_")[-4:-2]) for x in trained_ubm_dirs]
    policy_names = sorted(list(set(policy_names)))
    assert len(policy_names) == NUM_POLICIES_ANALYZED

    for policy_name in policy_names:
        for repetition in range(1, NUM_REPETITIONS + 1):
            encoder_distance_path = ENCODER_DISTANCE_MATRIX_TEMPLATE.format(num_traj=num_traj, env=env, policy_name=policy_name, repetition_num=repetition)
            # If already exists, skip extracting pivectors for this
            if os.path.isfile(encoder_distance_path):
                continue

            # Hacky thing to load up which trajectories were sampled.
            ubm_path = UBM_TEMPLATE.format(num_traj=num_traj, num_components=NUM_COMPONENTS, env=env, policy_name=policy_name, repetition_num=repetition)
            ubm_data = np.load(ubm_path)
            trajectory_indeces = ubm_data["trajectory_indeces"]
            ubm_data.close()

            # Load trajectory data
            trajectories_path = glob(os.path.join(TRAJECTORY_TEMPLATE.format(env=env, policy_name=policy_name), "*"))
            trajectories_path = sorted(trajectories_path)
            # Unlike previously, this will not be concatenated
            policy_datas = []
            all_average_episodic_returns = []
            for trajectory_i, trajectory_path in enumerate(trajectories_path):
                data = np.load(trajectory_path)
                keys = sorted(list(data.keys()))
                all_average_episodic_returns.append(data["episodic_rewards"].mean())
                # Take trajectories at same indeces as in used in training UBM.
                # First make sure it is in same order as with ubm training
                datas = [data[key] for key in keys if "traj" in key]
                datas = [datas[i] for i in trajectory_indeces[trajectory_i]]
                policy_datas.append(datas)

            num_pivectors = len(policy_datas)

            # Ravel all policy data for training
            all_data = []
            for policy_data in policy_datas:
                all_data.extend(policy_data)
            # Fun part: Train the encoder for trajectories
            encoder_network = train_trajectory_encoder(all_data)

            # Encode policies into distributions
            policy_encodings = [encode_policy_into_gaussian(encoder_network, policy_data) for policy_data in policy_datas]

            distance_matrix = np.ones((num_pivectors, num_pivectors))
            for i in range(num_pivectors):
                # Halve computation required
                for j in range(i, num_pivectors):
                    # Symmetric KL-divergence between the two policies, as in gaussian case
                    policy_i = policy_encodings[i]
                    policy_j = policy_encodings[j]
                    distance = None
                    with th.no_grad():
                        distance = th.distributions.kl_divergence(policy_i, policy_j) + th.distributions.kl_divergence(policy_j, policy_i)

                    distance_matrix[i, j] = distance.item()
                    distance_matrix[j, i] = distance.item()
            np.savez(
                encoder_distance_path,
                distance_matrix=distance_matrix,
                average_episodic_rewards=all_average_episodic_returns
            )


def compute_encoder_distances_worker_sugarcoat(kwargs):
    # A simple kwargs unpacker for compute_encoder_distances_worker
    compute_encoder_distance_worker(**kwargs)


def compute_encoder_distance_matrices():
    # We need to do heavy work here so parallelize in the
    # dirty way we did before
    jobs = []
    for num_traj_index, num_traj in enumerate(NUM_TRAJECTORIES):
        for env in ENVS:
            jobs.append(dict(
                num_traj_index=num_traj_index,
                num_traj=num_traj,
                env=env
            ))
    # Limit due to memory use
    workers = pool.Pool(4)
    for _ in tqdm(workers.imap(compute_encoder_distances_worker_sugarcoat, jobs), total=len(jobs), desc="encoders"):
        pass
    workers.close()


def compute_discretization_distance_worker(num_traj_index, num_traj, env):
    if "Bipedal" in env:
        print("[Warning] Skipping env {} in discrete binning due to high memory requirement".format(env))
        return

    # First get list of different policy names we have so we can iterate over them
    # (We are not actually using UBMs  here)
    trained_ubms = glob(UBM_TEMPLATE.format(num_traj=num_traj, num_components=NUM_COMPONENTS, env=env, policy_name="*", repetition_num="*"))
    trained_ubm_dirs = [os.path.basename(os.path.dirname(x)) for x in trained_ubms]
    policy_names = ["_".join(x.split("_")[-4:-2]) for x in trained_ubm_dirs]
    policy_names = sorted(list(set(policy_names)))
    assert len(policy_names) == NUM_POLICIES_ANALYZED

    for policy_name in policy_names:
        for repetition in range(1, NUM_REPETITIONS + 1):
            discretization_distance_path = DISCRETIZATION_DISTANCE_MATRIX_TEMPLATE.format(num_traj=num_traj, env=env, policy_name=policy_name, repetition_num=repetition)
            # If already exists, skip extracting pivectors for this
            if os.path.isfile(discretization_distance_path):
                continue

            # Hacky thing to load up which trajectories were sampled.
            ubm_path = UBM_TEMPLATE.format(num_traj=num_traj, num_components=NUM_COMPONENTS, env=env, policy_name=policy_name, repetition_num=repetition)
            ubm_data = np.load(ubm_path)
            trajectory_indeces = ubm_data["trajectory_indeces"]
            ubm_data.close()

            # Load trajectory data
            trajectories_path = glob(os.path.join(TRAJECTORY_TEMPLATE.format(env=env, policy_name=policy_name), "*"))
            trajectories_path = sorted(trajectories_path)
            # Unlike previously, this will not be concatenated
            policy_datas = []
            all_average_episodic_returns = []
            for trajectory_i, trajectory_path in enumerate(trajectories_path):
                data = np.load(trajectory_path)
                keys = sorted(list(data.keys()))
                all_average_episodic_returns.append(data["episodic_rewards"].mean())
                # Take trajectories at same indeces as in used in training UBM.
                # First make sure it is in same order as with ubm training
                datas = [data[key] for key in keys if "traj" in key]
                datas = [datas[i] for i in trajectory_indeces[trajectory_i]]
                policy_datas.append(np.concatenate(datas, axis=0))

            num_pivectors = len(policy_datas)

            # Ravel all policy data for binning operations
            all_data = np.concatenate(policy_datas, axis=0)

            # Add margin to both sides to make sure bins are in [0 ... NUM_BINS - 1]
            mins, maxs = np.min(all_data, axis=0) - 1e-4, np.max(all_data, axis=0) + 1e-4
            bin_sizes = (maxs - mins) / NUM_DISCRETE_BINS
            bin_matrix_shape = [NUM_DISCRETE_BINS] * bin_sizes.shape[0]
            ndim = bin_sizes.shape[0]
            # Free up memory
            del all_data

            def create_discrete_binning(policy_data, out):
                """Discretize policy data and store results in out"""
                discretized_data = np.floor((policy_data - mins) / bin_sizes).astype(np.uint)
                # Stackoverflow #2004364
                np.add.at(out, tuple([discretized_data[:, d] for d in range(ndim)]), 1)
                out /= len(policy_data)

            distance_matrix = np.ones((num_pivectors, num_pivectors))

            policy_i = np.ones(bin_matrix_shape, dtype=np.float64) * 0.0
            policy_j = np.ones(bin_matrix_shape, dtype=np.float64) * 0.0
            for i in range(num_pivectors):
                policy_i *= 0.0
                create_discrete_binning(policy_datas[i], policy_i)
                # Halve computation required
                for j in range(i, num_pivectors):
                    # Use total-variation divergence here as it is more comfortable
                    # compute (no logs, no naughty divisions by zero)
                    policy_j *= 0.0
                    create_discrete_binning(policy_datas[j], policy_j)

                    total_variation_distance = 0.5 * np.sum(np.abs(policy_i - policy_j))

                    # Above is symmetric
                    distance_matrix[i, j] = total_variation_distance
                    distance_matrix[j, i] = total_variation_distance

            np.savez(
                discretization_distance_path,
                distance_matrix=distance_matrix,
                average_episodic_rewards=all_average_episodic_returns
            )


def compute_discretization_distances_worker_sugarcoat(kwargs):
    # A simple kwargs unpacker for compute_discretization_distances_worker
    compute_discretization_distance_worker(**kwargs)


def compute_discretization_distance_matrices():
    # We need to do heavy work here so parallelize in the
    # dirty way we did before
    jobs = []
    for num_traj_index, num_traj in enumerate(NUM_TRAJECTORIES):
        for env in ENVS:
            jobs.append(dict(
                num_traj_index=num_traj_index,
                num_traj=num_traj,
                env=env
            ))
    # Limit due to memory use
    workers = pool.Pool(4)
    for _ in tqdm(workers.imap(compute_discretization_distances_worker_sugarcoat, jobs), total=len(jobs), desc="discretization"):
        pass
    workers.close()


def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    # Extract pivectors for each repetition+traj_num+component+policy+env
    # combination.
    extract_pivectors()
    # Extract single-gaussian BEs
    extract_gaussians()
    # Compute distance matrices for each pivector file.
    # Each file contains multiple pivectors and we
    # want to see if distances between them change between
    # runs
    compute_pivector_distance_matrices()
    compute_gaussian_distance_matrices()
    compute_discriminator_distance_matrices()
    compute_encoder_distance_matrices()
    compute_discretization_distance_matrices()


if __name__ == "__main__":
    main()
