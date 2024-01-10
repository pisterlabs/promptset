import gym
import torch
from torch import nn
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_action_dim
from models.feature_extractors import TransformerFeaturesExtractor
from models.distributions import (
    LatticeNoiseDistribution,
    LatticeAttentionNoiseDistribution,
    TransformerStateDependentNoiseDistribution,
    TransformerGaussianDistribution,
    LateNoiseDistribution
)
from stable_baselines3.common.distributions import (
    Distribution,
    DiagGaussianDistribution,
    CategoricalDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    BernoulliDistribution,
)
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from models.helpers import Mean
from models.extractors import TransformerExtractor
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from definitions import ROOT_DIR
import pickle
from datetime import datetime
import os


class MuscleTransformerPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde=False,
        log_std_init: float = 0.0,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_lattice=False,
        std_clip=(1e-3, 10),
        std_reg=0,
        **unused_kwargs,
    ):
        print("Warining: unused arguments", unused_kwargs)
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            TransformerFeaturesExtractor,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        if net_arch is None:
            # net_arch = [dict(pi=[64, 64], vf=[64, 64])]
            net_arch = []

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = TransformerFeaturesExtractor(
            self.observation_space, **self.features_extractor_kwargs
        )
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = True
        self.log_std_init = log_std_init
        # Keyword arguments for gSDE distribution
        if use_lattice:
            assert use_sde
            self.dist_kwargs = {
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
                "std_clip": std_clip,
                "std_reg": std_reg,
            }
            self.action_dist = LatticeAttentionNoiseDistribution(
                get_action_dim(action_space), **self.dist_kwargs
            )
        elif use_sde:
            assert not use_lattice
            self.dist_kwargs = {
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
                "std_reg": std_reg,
            }
            self.action_dist = TransformerStateDependentNoiseDistribution(
                get_action_dim(action_space), **self.dist_kwargs
            )
        else:
            self.dist_kwargs = None
            # self.action_dist = PerMuscleDiagGaussianDistribution(
            #     get_action_dim(action_space)
            # )
            self.action_dist = TransformerGaussianDistribution(
                action_dim=get_action_dim(action_space)
            )
        self.use_sde = use_sde
        self.use_lattice = use_lattice

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.mlp_extractor = TransformerExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, TransformerGaussianDistribution):
            (
                self.action_net,
                self.log_std_net,
                self.log_std,
            ) = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_dim_pi,
                log_std_init=self.log_std_init,
            )
            if isinstance(self.action_dist, TransformerStateDependentNoiseDistribution):
                self.log_std_scale_net = self.action_dist.log_std_scale_net

        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Sequential(
            Mean(dim=1), nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        )
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            raise NotImplementedError()

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        if isinstance(self.action_dist, TransformerGaussianDistribution):
            mean_actions = self.action_net(latent_pi)
            log_std_actions = self.log_std_net(
                latent_pi.detach()
            )  # Do not backpropagate the std branch of the network
            std_actions = log_std_actions.exp()
            std_actions = std_actions / std_actions.mean() * self.log_std.exp()
            return self.action_dist.proba_distribution(mean_actions, std_actions)
        else:
            return super()._get_action_dist_from_latent(latent_pi)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob


class LatticeRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        use_lattice=True,
        std_clip=(1e-3, 10),
        expln_eps=1e-6,
        std_reg=0,
        alpha=1,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if use_lattice:
            if self.use_sde:
                self.dist_kwargs.update(
                    {
                        "epsilon": expln_eps,
                        "std_clip": std_clip,
                        "std_reg": std_reg,
                        "alpha": alpha,
                    }
                )
                self.action_dist = LatticeNoiseDistribution(
                    get_action_dim(self.action_space), **self.dist_kwargs
                )
            else:
                self.action_dist = LateNoiseDistribution(get_action_dim(self.action_space), std_reg)
            self._build(lr_schedule)
            
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, _ = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)

        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
