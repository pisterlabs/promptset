from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

import numpy as np

import gym

import stable_baselines3 as sb
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
import math

class FixedStdActorCriticPolicy(sb.common.policies.ActorCriticPolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = [dict(pi=[256, 256], vf=[256, 256])],
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = math.log(1/20.0),
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):        
        super(FixedStdActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.log_std_action_bound_start = torch.tensor([math.log(1/10.0) for bound in action_space.high], device="cuda:0")
        self.log_std_action_bound_finish = torch.tensor([math.log(1/20.0) for bound in action_space.high], device="cuda:0")
        self.fix_action_net_stddev()

        return 

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 0.01,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        actor_params = list(self.action_net.parameters()) + list(self.mlp_extractor.parameters())
        self.optimizer_actor = self.optimizer_class(actor_params, lr=lr_schedule(1), **self.optimizer_kwargs)

        critic_params = list(self.value_net.parameters()) + list(self.features_extractor.parameters())
        self.optimizer_critic = self.optimizer_class(critic_params, lr=5e-4, **self.optimizer_kwargs)

    def fix_action_net_stddev(self) -> None: 
        self.log_std = nn.Parameter(self.log_std_action_bound_start, requires_grad=False)

    def update_action_noise(self, progress_remaining) -> None:
        if (progress_remaining > 0.5):
            progress_remaining_biased = 2*progress_remaining - 1
        else:
            progress_remaining_biased = 0.0


        self.log_std = nn.Parameter(torch.log(torch.exp(self.log_std_action_bound_start) * progress_remaining_biased 
                                            + torch.exp(self.log_std_action_bound_finish) * (1-progress_remaining_biased)) , requires_grad=False)



