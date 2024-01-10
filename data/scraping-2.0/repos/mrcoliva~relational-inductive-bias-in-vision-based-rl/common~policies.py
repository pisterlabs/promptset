from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple

import gym
import numpy as np
import torch as th

from torch import nn
from functools import partial
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from networks import ActionValueGraphNetwork
from utils.graph_builder import GraphBuilder

from stable_baselines3.common.distributions import (
    Distribution,
    DiagGaussianDistribution,
    StateDependentNoiseDistribution,
)

def activation_fn_from_string(fn_str: str) -> nn.Module:
    return {
        "tanh": nn.Tanh, 
        "relu": nn.ReLU, 
        "elu": nn.ELU, 
        "leaky_relu": nn.LeakyReLU
    }[fn_str]

class GraphNetworkPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs):
        
        self.config = kwargs.pop('config')
        self.obs_mode = self.config['obs_mode']
        self.use_input_models = self.config['use_input_models']
        self.global_input_model_latent_dim = self.config['global_input_model_latent_dim']
        self.joint_input_model_latent_dim = self.config['joint_input_model_latent_dim']
        self.action_dim = get_action_dim(action_space)

        self.numeric_features_dim = kwargs.pop('numeric_features_dim')
        self.global_feature_dim = kwargs.pop('global_feature_dim')
        self.joint_feature_dim = kwargs.pop('joint_feature_dim')
        graph_builder_class = kwargs.pop('graph_builder_class', GraphBuilder)

        sf = self.global_input_model_latent_dim if self.use_input_models else self.global_feature_dim
        nf = self.joint_input_model_latent_dim if self.use_input_models else self.joint_feature_dim

        self.graph_builder = graph_builder_class(
            n_nodes=self.action_dim,
            n_shared_features=sf,
            n_features_per_node=nf,
            add_forward_skip_connections=self.config['graph_forward_connections'],
            add_end_effector_connections=self.config['graph_ee_back_connections'],
            add_self_edges=self.config['graph_self_edges']
        )

        # for some reason the super constructor always overrides activation_fn to nn.Tanh
        # workaround: use a different variable
        self.actual_activation_fn = activation_fn_from_string(kwargs.pop('activation_fn'))
        
        super(GraphNetworkPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            self.actual_activation_fn,
            *args,
            **kwargs,
        )
    
    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        
        if self.use_input_models:
            self._build_input_models()

        if self.config['actor_global_avg_pool']:
            self.action_net = nn.Linear(self.config['gn_hidden_dim'], self.action_dim)
        else:
            self.node_action_nets = nn.ModuleList([
                nn.Linear(self.config['gn_hidden_dim'], 1) for _ in range(self.action_dim)
            ])

        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }

            try:
                for net in self.node_action_nets:
                    module_gains[net] = 0.01
            except:
                pass
            
            if self.use_input_models:
                module_gains[self.shared_input_model] = np.sqrt(2)
                module_gains[self.joint_input_model] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    
    def _build_input_models(self):
        self.shared_input_model = nn.Sequential(
            nn.Linear(
                self.global_feature_dim, 
                self.global_input_model_latent_dim), 
            nn.Tanh()
        ).to(self.device)
        
        self.joint_input_model = nn.Sequential(
            nn.Linear(
                self.joint_feature_dim,
                self.joint_input_model_latent_dim),
            nn.Tanh()
        ).to(self.device)

    def _build_mlp_extractor(self) -> None:
        if self.config['image_only']:
            input_dim = self.global_input_model_latent_dim
        else:
            input_dim = self.graph_builder.sample_dim

        self.mlp_extractor = ActionValueGraphNetwork(
            config=self.config,
            graph_builder=self.graph_builder,
            input_dim=input_dim,
            activation_fn=self.actual_activation_fn,
            device=self.device)

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """Preprocess the observation if needed and extract features."""
        assert self.features_extractor is not None, "No features extractor was set"

        obs = obs.float()

        if self.obs_mode == 'numeric':
            return self.features_extractor(obs)

        # convert to channel-first
        obs = obs.permute(0, 3, 1, 2)

        if self.obs_mode == 'images':
            return self.features_extractor(obs)
        
        elif self.obs_mode == 'combined':
            assert obs.shape[1] == 4, f'Invalid channel dimension: {obs.shape}'

            images = obs[:, 0:3, :, :]
            _, global_features = self.features_extractor(images, only_latent=True)

            if self.config['image_only']:
                return global_features

            offset = 2 # self.global_feature_dim
            joint_features = obs[:, 3, offset:self.numeric_features_dim + offset, 0]
            joint_features = joint_features.reshape((-1, self.numeric_features_dim))
            
            return th.cat([global_features, joint_features], dim=1)
        else:
            raise ValueError('Invalid obs_mode `{self.obs_mode}`')

    def _apply_input_models(self, obs: th.Tensor) -> th.Tensor:
        batch_size = obs.shape[0]

        x_global = obs[:, :self.global_feature_dim]
        x_global = self.shared_input_model(x_global)

        if self.config['image_only']:
            return x_global
        
        x_joints = obs[:, self.global_feature_dim:]\
            .reshape(batch_size, self.action_dim, self.joint_feature_dim)
        
        x_joints = self.joint_input_model(x_joints).reshape(batch_size, -1)
        return th.cat([x_global, x_joints], dim=1)

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.
        :param obs: Observation
        :return: Latent codes for the actor, the value function and for gSDE function
        """
        features = self.extract_features(obs)

        if self.use_input_models:
            features = self._apply_input_models(features)
        
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde
    
    def _get_action_dist_from_latent(
        self, 
        latent_pi: th.Tensor, 
        latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        if self.config['actor_global_avg_pool']:
            mean_actions = self.action_net(latent_pi)
        else:
            mean_actions = th.zeros(latent_pi.shape[0], self.action_dim)
            
            for i in range(self.action_dim):
                action = self.node_action_nets[i](latent_pi[:, i]).reshape(latent_pi.shape[0])
                mean_actions[:, i] = action

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

   