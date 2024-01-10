from argparse import ArgumentParser
import pickle

import torch as th
import numpy as np
import gym
from stable_baselines3 import PPO
from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import (
    CnnRewardNet,
    NormalizedRewardNet,
)
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
import minerl  # noqa: F401
from openai_vpt.agent import MineRLAgent

import sb3_minerl_envs  # noqa: F401
from sb3_policy_wrapper import MinecraftActorCriticPolicy
from gym_wrappers import ObservationToInfos


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def preference_based_RL_train(env_str, in_model, in_weights, out_weights):

    # Setup MineRL environment
    minerl_env_str = "MineRLBasalt" + env_str
    env = gym.make(minerl_env_str + "-v0")

    # Setup MineRL agent
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    minerl_agent = MineRLAgent(
        env,
        device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    minerl_agent.load_weights(in_weights)

    # Freeze most params if using small dataset
    for param in minerl_agent.policy.parameters():
        param.requires_grad = False
    # Unfreeze final layers and policy and value head
    for param in minerl_agent.policy.net.lastlayer.parameters():
        param.requires_grad = True
    for param in minerl_agent.policy.pi_head.parameters():
        param.requires_grad = True
    for param in minerl_agent.policy.value_head.parameters():
        param.requires_grad = True

    # Setup MineRL VecEnv
    venv = make_vec_env(
        minerl_env_str + "SB3-v0",
        # Keep this at 1 since we are not keeping track of multiple hidden states
        n_envs=1,
        # This should be sufficiently high for the given task
        max_episode_steps=20,
        env_make_kwargs={"minerl_agent": minerl_agent},
    )

    # Setup preference-based reinforcement learning using the imitation package
    # TODO In general, check whether algorithm hyperparams make sense and tune

    # TODO reuse ImpalaCNN from the VPT models with a regression head
    image_obs_space = gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
    reward_net = CnnRewardNet(image_obs_space, venv.action_space, use_action=False)
    normalized_reward_net = NormalizedRewardNet(reward_net, RunningNorm)
    preference_model = preference_comparisons.PreferenceModel(normalized_reward_net)

    # TODO design more useful fragmenter for MineRL trajcetories,
    # e.g. only compare last parts of episodes
    fragmenter = preference_comparisons.MineRLFragmenter(warning_threshold=0, seed=0)

    gatherer = preference_comparisons.PrefCollectGatherer(
        pref_collect_address="http://127.0.0.1:8000",
        video_output_dir="/home/aicrowd/pref-collect/videofiles/",
    )

    # TODO imitation also provides EnsembleTrainer
    # (which requires a RewardEnsemble), which trainer should we use?
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        model=normalized_reward_net,
        loss=preference_comparisons.CrossEntropyRewardLoss(preference_model),
        epochs=3,
    )

    agent = PPO(
        policy=MinecraftActorCriticPolicy,
        policy_kwargs={
            "minerl_agent": minerl_agent,
            "optimizer_class": th.optim.Adam,
        },
        env=venv,
        seed=0,
        n_steps=512 // venv.num_envs,
        batch_size=32,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
    )

    trajectory_generator = preference_comparisons.MineRLAgentTrainer(
        algorithm=agent,
        reward_fn=reward_net,
        venv=venv,
        exploration_frac=0.0,
        seed=0,
    )

    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations=5,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        fragment_length=10,
        transition_oversampling=1,
        initial_comparison_frac=0.1,
        allow_variable_horizon=True,
        seed=0,
        initial_epoch_multiplier=1,
    )

    # Run training
    pref_comparisons.train(
        total_timesteps=500,  # For good performance this should be 1_000_000
        total_comparisons=10,  # For good performance this should be 5_000
    )

    venv.close()

    # Save finetuned weights
    state_dict = minerl_agent.policy.state_dict()
    th.save(state_dict, out_weights)

    print("Finished")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        help="Environment name from [FindCave, MakeWaterfall, \
            CreateVillageAnimalPen, BuildVillageHouse]",
        default="FindCave",
    )
    parser.add_argument(
        "--in-model",
        type=str,
        help="Path to the .model file to be finetuned",
        default="data/VPT-models/foundation-model-1x.model",
    )
    parser.add_argument(
        "--in-weights",
        type=str,
        help="Path to the .weights file to be finetuned",
        default="data/VPT-models/foundation-model-1x.weights",
    )
    parser.add_argument(
        "--out-weights",
        type=str,
        help="Path where finetuned weights will be saved",
        default="train/PrefRLFinetuned.weights",
    )

    args = parser.parse_args()
    preference_based_RL_train(
        args.env, args.in_model, args.in_weights, args.out_weights
    )
