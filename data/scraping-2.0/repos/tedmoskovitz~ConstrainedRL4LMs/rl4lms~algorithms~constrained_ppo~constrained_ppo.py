import warnings
from typing import Any, Dict, Optional, Type, Union
import pdb

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.policy.base_policy import EvaluateActionsOutput


class ConstrainedPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param constraint_vf_coef: Constraint value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        tracker: Tracker,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        constraint_vf_coef: float = 0.5,
        kl_vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        constraint_threshold: float = 0.1,
        squash_lagrange: bool = True,
        lagrange_lr: float = 1e-2,
        lagrange_init: Optional[float] = None,
        fixed_lagrange: bool = False,
        maximizing_reward: str = "kl",
        task_threshold: float = 0.1,
        equality_constraints: bool = False,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.constraint_vf_coef = constraint_vf_coef
        self.kl_vf_coef = kl_vf_coef
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self._tracker = tracker
        self.constraint_threshold = constraint_threshold
        self.squash_lagrange = squash_lagrange
        self.equality_constraints = equality_constraints
        if self.squash_lagrange and self.equality_constraints:
            self.squash_fn = th.tanh
        else:
            self.squash_fn = th.sigmoid
        if lagrange_init is None:
            if squash_lagrange and equality_constraints:
                lagrange_init = 0.0
            elif squash_lagrange:
                lagrange_init = 0.0
            else:
                lagrange_init = 0.5
        self.maximizing_reward = maximizing_reward.lower()
        if self.maximizing_reward not in ['kl', 'task', 'all']:
            raise ValueError("maximizing_reward must be one of ['kl', 'task', 'all']")
        
        if self.maximizing_reward in ['kl', 'all']:
            lagrange_init = [lagrange_init] * 2
        self.lagrange = th.tensor(
            lagrange_init, requires_grad=True, device=self.device, dtype=th.float32)
        self.lagrange_optimizer = th.optim.SGD([self.lagrange], lr=lagrange_lr, momentum=0.1)
        self.fixed_lagrange = fixed_lagrange
        self.task_threshold = task_threshold


        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining)

        entropy_losses = []
        pg_losses, task_value_losses, constraint_value_losses = [], [], []
        kl_value_losses, actual_constraint_returns_list = [], []
        lagrange_losses = []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for batch_ix, rollout_data in enumerate(list(self.rollout_buffer.get(self.batch_size))):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                evaluation_output: EvaluateActionsOutput = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values, log_prob, entropy = evaluation_output.values, evaluation_output.log_prob, evaluation_output.entropy
                task_values = values[..., 0].flatten()
                constraint_values = values[..., 1].flatten()
                kl_values = values[..., 2].flatten()
                # Normalize advantage
                task_advantages = rollout_data.task_advantages
                constraint_advantages = rollout_data.constraint_advantages
                kl_advantages = rollout_data.kl_advantages
                if self.normalize_advantage:
                    task_advantages = (task_advantages - task_advantages.mean()
                                  ) / (task_advantages.std() + 1e-8)
                    constraint_advantages = (constraint_advantages - constraint_advantages.mean()
                                    ) / (constraint_advantages.std() + 1e-8)
                    kl_advantages = (kl_advantages - kl_advantages.mean()
                                    ) / (kl_advantages.std() + 1e-8)
                    
                    
                # compute mixed advantages
                if self.maximizing_reward == 'kl':
                    if self.squash_lagrange:
                        lagrange = self.squash_fn(self.lagrange)
                        mixed_advantages = (2 - lagrange.sum()) * kl_advantages + lagrange[0] * task_advantages + lagrange[1] * constraint_advantages
                    else:
                        mixed_advantages = kl_advantages * self.lagrange[0] * task_advantages + self.lagrange[1] * constraint_advantages
                elif self.maximizing_reward == 'all':
                    if self.squash_lagrange:
                        lagrange = self.squash_fn(self.lagrange)
                        total_advantages = task_advantages + constraint_advantages
                        mixed_advantages = (2 - lagrange.sum()) * total_advantages - lagrange[0] * task_advantages - lagrange[1] * constraint_advantages 
                    else:
                        total_advantages = task_advantages + constraint_advantages
                        mixed_advantages = total_advantages - self.lagrange[0] * task_advantages - self.lagrange[1] * constraint_advantages
                else:
                    if self.squash_lagrange:
                        lagrange = self.squash_fn(self.lagrange)
                        mixed_advantages = (1 - lagrange) * task_advantages + lagrange * constraint_advantages
                    else:
                        mixed_advantages = task_advantages + self.lagrange * constraint_advantages


                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = mixed_advantages * ratio
                policy_loss_2 = mixed_advantages * \
                    th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    task_values_pred = task_values
                    constraint_values_pred = constraint_values
                    kl_values_pred = kl_values
                else:
                    # Clip the different between old and new value
                    task_values_pred = rollout_data.old_task_values + th.clamp(
                        task_values - rollout_data.old_task_values, -clip_range_vf, clip_range_vf
                    )
                    constraint_values_pred = rollout_data.old_constraint_values + th.clamp(
                        constraint_values - rollout_data.old_constraint_values, -clip_range_vf, clip_range_vf
                    )
                    kl_values_pred = rollout_data.old_kl_values + th.clamp(
                        kl_values - rollout_data.old_kl_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                task_value_loss = F.mse_loss(rollout_data.task_returns, task_values_pred)
                task_value_losses.append(task_value_loss.item())
                constraint_value_loss = F.mse_loss(rollout_data.constraint_returns, constraint_values_pred)
                constraint_value_losses.append(constraint_value_loss.item())
                kl_value_loss = F.mse_loss(rollout_data.kl_returns, kl_values_pred)
                kl_value_losses.append(kl_value_loss.item())
                value_loss = self.vf_coef * task_value_loss + self.constraint_vf_coef * constraint_value_loss
                value_loss += self.kl_vf_coef * kl_value_loss

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + value_loss

                # we need to cancel out the kl returns so that the constraint is only over the
                # actual constraint reward function
                # constraint_return = actual_constraint_return + kl_return
                # if self.maximize_kl_reward: # TODO: this is a hack, need to re-name/clean-up
                if self.maximizing_reward == 'kl':
                    # [batch_size,]
                    constraint_violations = rollout_data.ep_constraint_reward_togo.mean() - self.constraint_threshold
                    # [batch_size,]
                    task_violations = rollout_data.ep_task_reward_togo.mean() - self.task_threshold
                    # [n_constriants,]
                    lagrange = self.squash_fn(self.lagrange) if self.squash_lagrange else self.lagrange
                    lagrange_loss = lagrange[0] * task_violations + lagrange[1] * constraint_violations
                elif self.maximizing_reward == 'all':
                    actual_constraint_returns = rollout_data.ep_constraint_reward_togo - rollout_data.ep_kl_reward_togo
                    actual_task_returns = rollout_data.ep_task_reward_togo - rollout_data.ep_kl_reward_togo
                    constraint_violations = self.constraint_threshold - actual_constraint_returns.mean()
                    task_violations = self.task_threshold - actual_task_returns.mean()
                    lagrange = self.squash_fn(self.lagrange) if self.squash_lagrange else self.lagrange
                    lagrange_loss = lagrange[0] * task_violations + lagrange[1] * constraint_violations
                else:
                    actual_constraint_returns = rollout_data.constraint_returns - rollout_data.kl_returns
                    constraint_violations = actual_constraint_returns.mean() - self.constraint_threshold
                    lagrange = self.squash_fn(self.lagrange) if self.squash_lagrange else self.lagrange
                    lagrange_loss = lagrange * constraint_violations
                    actual_constraint_returns_list.append(actual_constraint_returns.mean().item())
                lagrange_losses.append(lagrange_loss.item())

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # Lagrange multiplier update
                if not self.fixed_lagrange:
                    self.lagrange_optimizer.zero_grad()
                    lagrange_loss.backward()
                    th.nn.utils.clip_grad_norm_(
                        [self.lagrange], self.max_grad_norm)
                    self.lagrange_optimizer.step()
                    if not self.squash_lagrange and not self.equality_constraints:
                        self.lagrange.data = th.clamp(self.lagrange.data, min=0)


            if not continue_training:
                break

        self._n_updates += self.n_epochs
        task_explained_var = explained_variance(
            self.rollout_buffer.task_values.flatten(), self.rollout_buffer.task_returns.flatten())
        constraint_explained_var = explained_variance(
            self.rollout_buffer.constraint_values.flatten(), self.rollout_buffer.constraint_returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/task_value_loss", np.mean(task_value_losses))
        self.logger.record("train/constraint_value_loss", np.mean(constraint_value_losses))
        self.logger.record("train/kl_value_loss", np.mean(kl_value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/task_explained_variance", task_explained_var)
        self.logger.record("train/constraint_explained_variance", constraint_explained_var)
        lagrange = self.squash_fn(self.lagrange) if self.squash_lagrange else self.lagrange
        # self.logger.record("train/lagrange", lagrange.item())
        self.logger.record("train/lagrange_loss", np.mean(lagrange_losses))
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates",
                           self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        train_info = {
            "ppo/entropy_loss":  np.mean(entropy_losses).item(),
            "ppo/policy_gradient_loss": np.mean(pg_losses).item(),
            "ppo/value_loss": np.mean(task_value_losses).item(),
            "ppo/constraint_value_loss": np.mean(constraint_value_losses).item(),
            "ppo/kl_value_loss": np.mean(kl_value_losses).item(),
            "ppo/approx_kl": np.mean(approx_kl_divs).item(),
            "ppo/explained_variance": task_explained_var,
            "ppo/constraint_explained_variance": constraint_explained_var,
            "ppo/lagrange_loss": np.mean(lagrange_losses),
            "ppo/constraint_violations": constraint_violations.item(),
            "ppo/constraint_returns": rollout_data.constraint_returns.mean().item(),
            "ppo/task_returns": rollout_data.task_returns.mean().item(),
            "ppo/kl_returns": rollout_data.kl_returns.mean().item(),
            "ppo/actual_constraint_returns": np.mean(actual_constraint_returns_list),
            "ppo/ep_constraint_reward_togo": rollout_data.ep_constraint_reward_togo.mean().item(),
            "ppo/ep_task_reward_togo": rollout_data.ep_task_reward_togo.mean().item(),
            "ppo/ep_kl_reward_togo": rollout_data.ep_kl_reward_togo.mean().item(),
            "ppo/task_threshold": self.task_threshold,
            "ppo/constraint_threshold": self.constraint_threshold,
        }
        # if self.maximize_kl_reward:
        if self.maximizing_reward in ['kl', 'all']:
            train_info.update({"ppo/task_lagrange": lagrange[0].item(),
                               "ppo/constraint_lagrange": lagrange[1].item(),
                               "ppo/task_violations": task_violations.item()})
        else:
            train_info.update({"ppo/lagrange": lagrange.item(),})

        self._tracker.log_training_infos(train_info)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "ConstrainedPPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "ConstrainedPPO":

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
