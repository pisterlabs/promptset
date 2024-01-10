"""Contributed port of MADDPG from OpenAI baselines.

The implementation has a couple assumptions:
- The number of agents is fixed and known upfront.
- Each agent is bound to a policy of the same name.
- Discrete actions are sent as logits (pre-softmax).

For a minimal example, see rllib/examples/two_step_game.py,
and the README for how to run with the multi-agent particle envs.
"""

import numpy as np
from gym.spaces import Box, Discrete
import logging
from typing import Optional, Type

from ray.rllib.agents.trainer import COMMON_CONFIG, with_common_config
from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer
from maddpg_tf_policy import MADDPGTFPolicy
from maddpg_torch_policy import MADDPGTorchPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils import merge_dicts

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # === Framework to run the algorithm ===
    "framework": "tf",

    # === Settings for each individual policy ===
    # ID of the agent controlled by this policy
    "agent_id": None,
    # Use a local critic for this policy.
    "use_local_critic": False,

    # === Evaluation ===
    # Evaluation interval
    "evaluation_interval": None,
    # Number of episodes to run per evaluation period.
    "evaluation_num_episodes": 10,

    # === Model ===
    # Apply a state preprocessor with spec given by the "model" config option
    # (like other RL algorithms). This is mostly useful if you have a weird
    # observation shape, like an image. Disabled by default.
    "use_state_preprocessor": False,
    # Postprocess the policy network model output with these hidden layers. If
    # use_state_preprocessor is False, then these will be the *only* hidden
    # layers in the network.
    "actor_hiddens": [64, 64],
    # Hidden layers activation of the postprocessing stage of the policy
    # network
    "actor_hidden_activation": "relu",
    # Postprocess the critic network model output with these hidden layers;
    # again, if use_state_preprocessor is True, then the state will be
    # preprocessed by the model specified with the "model" config option first.
    "critic_hiddens": [64, 64],
    # Hidden layers activation of the postprocessing state of the critic.
    "critic_hidden_activation": "relu",
    # N-step Q learning
    "n_step": 1,
    # Algorithm for good policies.
    "good_policy": "maddpg",
    # Algorithm for adversary policies.
    "adv_policy": "maddpg",
    # list of other agent_ids and policies to approximate (See MADDPG Section 4.2)
    "learn_other_policies": None,

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": int(1e6),
    # Observation compression. Note that compression makes simulation slow in
    # MPE.
    "compress_observations": False,
    # If set, this will fix the ratio of replayed from a buffer and learned on
    # timesteps to sampled from an environment and stored in the replay buffer
    # timesteps. Otherwise, the replay will proceed at the native ratio
    # determined by (train_batch_size / rollout_fragment_length).
    "training_intensity": None,
    # Force lockstep replay mode for MADDPG.
    "multiagent": merge_dicts(COMMON_CONFIG["multiagent"], {
        "replay_mode": "lockstep",
    }),

    # === Exploration ===
    "exploration_config": {
        "type": "GaussianNoise",
        # For how many timesteps should we return completely random actions,
        # before we start adding (scaled) noise?
        "random_timesteps": 1000,
        # The stddev (sigma) to be used for the actions
        "stddev": 0.5,
        # The initial noise scaling factor.
        "initial_scale": 1.0,
        # The final noise scaling factor.
        "final_scale": 0.02,
        # Timesteps over which to anneal scale (from initial to final values).
        "scale_timesteps": 10000,
    },
    # Extra configuration that disables exploration.
    "evaluation_config": {
        "explore": False
    },

    # === Optimization ===
    # Learning rate for the critic (Q-function) optimizer.
    "critic_lr": 1e-2,
    # Learning rate for the actor (policy) optimizer.
    "actor_lr": 1e-2,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 0,
    # Update the target by \tau * policy + (1-\tau) * target_policy
    "tau": 0.01,
    # Weights for feature regularization for the actor
    "actor_feature_reg": 0.001,
    # If not None, clip gradients during optimization at this value
    "grad_clip": 100,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 1024 * 25,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    "rollout_fragment_length": 100,
    # Size of a batched sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 1024,
    # Number of env steps to optimize for before returning
    "timesteps_per_iteration": 1000,

    # torch-specific model configs
    "twin_q": False,
    # delayed policy update
    "policy_delay": 1,
    # target policy smoothing
    # (this also replaces OU exploration noise with IID Gaussian exploration noise, for now)
    "smooth_target_policy": False,
    "use_huber": False,
    "huber_threshold": 1.0,
    "l2_reg": None,

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you're using the Async or Ape-X optimizers.
    "num_workers": 1,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 0,
})
# __sphinx_doc_end__
# yapf: enable


def _make_continuous_space(space):
    if isinstance(space, Box):
        return space
    elif isinstance(space, Discrete):
        return Box(low=np.zeros((space.n,)), high=np.ones((space.n,)))
    else:
        raise UnsupportedSpaceException("Space {} is not supported.".format(space))


def before_learn_on_batch(
    multi_agent_batch, policies, train_batch_size, framework="tf"
):
    # TODO: This should only operate on agents following maddpg, not ddpg!
    samples = {}

    # Modify keys.
    for pid, p in policies.items():
        i = p.config["agent_id"]
        keys = multi_agent_batch.policy_batches[pid].keys()
        keys = ["_".join([k, str(i)]) for k in keys]
        samples.update(dict(zip(keys, multi_agent_batch.policy_batches[pid].values())))

    # Make ops and feed_dict to get "new_obs" from target action sampler.
    new_obs_n = list()
    new_act_n = list()
    for k, v in samples.items():
        if "new_obs" in k:
            new_obs_n.append(v)

    if framework == "torch":

        def sampler(policy, obs):
            return policy.compute_actions(obs)[0]

        new_act_n = [
            sampler(policy, obs) for policy, obs in zip(policies.values(), new_obs_n)
        ]
    else:
        target_act_sampler_n = [p.target_act_sampler for p in policies.values()]
        new_obs_ph_n = [p.new_obs_ph for p in policies.values()]
        feed_dict = dict(zip(new_obs_ph_n, new_obs_n))
        new_act_n = p.sess.run(target_act_sampler_n, feed_dict)

    samples.update(
        {"new_actions_%d" % i: new_act for i, new_act in enumerate(new_act_n)}
    )

    # Share samples among agents.
    policy_batches = {pid: SampleBatch(samples) for pid in policies.keys()}
    return MultiAgentBatch(policy_batches, train_batch_size)


def add_maddpg_postprocessing(config):
    """Add the before learn on batch hook.

    This hook is called explicitly prior to TrainOneStep() in the execution
    setups for DQN and APEX.
    """

    def f(batch, workers, config):
        policies = dict(
            workers.local_worker().foreach_trainable_policy(lambda p, i: (i, p))
        )
        return before_learn_on_batch(
            batch, policies, config["train_batch_size"], config["framework"]
        )

    config["before_learn_on_batch"] = f
    return config


def get_policy_class(config: TrainerConfigDict) -> Optional[Type[Policy]]:
    """Policy class picker function. Class is chosen based on DL-framework.
    Args:
        config (TrainerConfigDict): The trainer's configuration dict.
    Returns:
        Optional[Type[Policy]]: The Policy class to use with PGTrainer.
            If None, use `default_policy` provided in build_trainer().
    """
    if config["framework"] == "torch":
        return MADDPGTorchPolicy
    else:
        return MADDPGTFPolicy


MADDPGTrainer = GenericOffPolicyTrainer.with_updates(
    name="MADDPG",
    default_config=DEFAULT_CONFIG,
    default_policy=MADDPGTFPolicy,
    get_policy_class=get_policy_class,
    validate_config=add_maddpg_postprocessing,
)
