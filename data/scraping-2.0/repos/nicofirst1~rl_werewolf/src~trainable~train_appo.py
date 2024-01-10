from ray.rllib.agents.ppo.appo_tf_policy import AsyncPPOTFPolicy

from policies import *
from utils import Params

_ = Params()

import logging

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from callbacks import  CustomCallbacks
from envs import CONFIGS
from models import ParametricActionsModel
from other.custom_utils import trial_name_creator
from policies.RandomTarget import RandomTarget
from wrappers import EvaluationWrapper


def mapping_static(agent_id):
    if "wolf" in agent_id:
        return "wolf_p_static"
    elif "vil" in agent_id:
        return "vill_p"
    else:
        raise NotImplementedError(f"Policy for role {agent_id} not implemented")


def mapping_dynamic(agent_id):
    if "wolf" in agent_id:
        return "wolf_p"
    elif "vil" in agent_id:
        return "vill_p"
    else:
        raise NotImplementedError(f"Policy for role {agent_id} not implemented")

if __name__ == '__main__':


    _ = ParametricActionsModel
    ray.init(local_mode=Params.debug, logging_level=logging.DEBUG, num_gpus=1)

    env_configs = CONFIGS

    env = EvaluationWrapper(env_configs)

    # define policies
    vill_p = (AsyncPPOTFPolicy, env.observation_space, env.action_space, {})
    ww_p = (RandomTarget, env.observation_space, env.action_space, {})

    policies = dict(
        wolf_p_static=ww_p,
        wolf_p=vill_p,
        vill_p=vill_p,
    )

    configs = {
        "env": EvaluationWrapper,
        "env_config": env_configs,
        "eager": False ,
        "eager_tracing": False,
        "num_workers": Params.n_workers,
        "num_gpus": Params.n_gpus,
        "batch_mode": "complete_episodes",
        "train_batch_size": 500,
        "rollout_fragment_length": 100,

        # PPO parameter taken from OpenAi paper
        "lr": 3e-4,
        "lambda": .95,
        "gamma": .998,
        "num_sgd_iter": 1, #default 1
        "replay_proportion":0.05,



        "callbacks": CustomCallbacks,

        # model configs
        "model": {
            "use_lstm": True,
            "custom_model": "pa_model",  # using custom parametric action model
        },
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": mapping_static,
            "policies_to_train": "vill_p"

        },

    }

    analysis = tune.run(
        APPOTrainer,
        local_dir=Params.RAY_DIR,
        config=configs,
        trial_name_creator=trial_name_creator,
        checkpoint_freq=Params.checkpoint_freq,
        keep_checkpoints_num=Params.max_checkpoint_keep,
        resume=Params.resume_training,
        reuse_actors=True

    )
