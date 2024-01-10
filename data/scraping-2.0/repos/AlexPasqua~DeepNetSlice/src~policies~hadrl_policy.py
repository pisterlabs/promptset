from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, Tuple

import gym
import networkx as nx
import numpy as np
import torch as th
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from .features_extractors import GCNsFeaturesExtractor
from .mlp_extractors.hadrl_mlp_extractor import HADRLActorCriticNet


class HADRLPolicy(MultiInputActorCriticPolicy):
    """ Policy network from the paper HA-DRL [1]

    [1] https://ieeexplore.ieee.org/document/9632824
    """
    name = 'HADRL Policy'

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            psn: nx.Graph,
            servers_map_idx_id: Dict[int, int],
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            gcn_layers_dims: Tuple[int] = (60,),
            nspr_out_features: int = 4,
            use_heuristic: bool = False,
            heu_kwargs: dict = None,
            *args,
            **kwargs,
    ):
        """
        :param observation_space: Observation space of the agent
        :param action_space: Action space of the agent
        :param lr_schedule: Learning rate schedule
        :param psn: Physical Service Network
        :param servers_map_idx_id: Mapping between servers' indexes and their IDs
        :param net_arch: architecture of the policy and value networks after the feature extractor
        :param activation_fn: Activation function
        :param gcn_layers_dims: Dimensions of the GCN layers
        :param nspr_out_features: Number of output features of the NSPR state
        :param use_heuristic: Whether to use the heuristic or not
        :param heu_kwargs: Keyword arguments for the heuristic
        """

        # assert len(net_arch) == 1 and isinstance(net_arch[0], dict), \
        #     "This policy allows net_arch to be a list with only one dict"

        self.psn = psn
        self.gcn_layers_dims = gcn_layers_dims  # saved in an attribute for logging purposes
        self.servers_map_idx_id = servers_map_idx_id
        self.use_heuristic = use_heuristic
        self.heu_kwargs = heu_kwargs

        super(HADRLPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # non-shared features extractors for the actor and the critic
        self.policy_features_extractor = GCNsFeaturesExtractor(
            observation_space, psn, nn.Tanh, gcn_layers_dims,
            nspr_out_features
        )
        self.value_features_extractor = GCNsFeaturesExtractor(
            observation_space, psn, nn.ReLU, gcn_layers_dims,
            nspr_out_features
        )
        self.features_dim = {'pi': self.policy_features_extractor.features_dim,
                             'vf': self.value_features_extractor.features_dim}
        delattr(self, "features_extractor")  # remove the shared features extractor

        # TODO: check what this step actually does
        # Disable orthogonal initialization
        # self.ortho_init = False

        # Workaround alert!
        # This method is called in the super-constructor. It creates the optimizer,
        # but using also the params of the features extractor before creating
        # our own 2 separate ones ('policy_features_extractor' and
        # 'value_features_extractor'). Therefore we need to re-create the optimizer
        # using the params of the correct new features extractor.
        # (it will also re-do a bunch of things like re-creating the mlp_extractor,
        # which was fine, but it's not a problem).
        self._rebuild(lr_schedule)

    def _rebuild(self, lr_schedule: Schedule) -> None:
        """
        Like method _build, but needed to be re-called to re-create the
        optimizer, since it was created using obsolete parameters, i.e. params
        including the ones of the default shared features extractor and NOT
        including the ones of the new features extractors.
        The mlp_extractor is recreated too, since it was created with incorrect features_dim.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        # action_net and value_net as created in the '_build' method are OK,
        # no need to recreate them.

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.policy_features_extractor: np.sqrt(2),
                self.value_features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HADRLActorCriticNet(
            action_space=self.action_space,
            psn=self.psn,
            net_arch=self.net_arch,
            servers_map_idx_id=self.servers_map_idx_id,
            features_dim=self.features_dim,
            use_heuristic=self.use_heuristic,
            heu_kwargs=self.heu_kwargs
        )

    def extract_features(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :return: the output of the feature extractor(s)
        """
        assert self.policy_features_extractor is not None and \
               self.value_features_extractor is not None
        preprocessed_obs = preprocess_obs(obs, self.observation_space,
                                          normalize_images=self.normalize_images)
        policy_features = self.policy_features_extractor(preprocessed_obs)
        value_features = self.value_features_extractor(preprocessed_obs)
        return policy_features, value_features

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> \
            Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        policy_features, value_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(policy_features, obs)
        latent_vf = self.mlp_extractor.forward_critic(value_features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> \
            Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        policy_features, value_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(policy_features, obs)
        latent_vf = self.mlp_extractor.forward_critic(value_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :return: the action distribution.
        """
        policy_features, _ = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(policy_features, obs)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        _, value_features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(value_features)
        return self.value_net(latent_vf)
