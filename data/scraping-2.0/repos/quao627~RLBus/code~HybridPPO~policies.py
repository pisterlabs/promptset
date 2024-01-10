"""Policies: abstract base class and concrete implementations."""

import collections
import copy
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from itertools import zip_longest

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import *
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from stable_baselines3.common.policies import BasePolicy

from sb3_contrib.common.maskable.distributions import MaskableDistribution, make_masked_proba_distribution

class HybridMlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous feature extractor (i.e. a CNN) or directly
    the observations (if no feature extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.
    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:
    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.
    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].
    Adapted from Stable Baselines.
    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        shared_net, policy_net_h, policy_net_l, value_net = [], [], [], []
        policy_only_layers_h = []  # Layer sizes of the network that only belongs to the policy network (high level)
        policy_only_layers_l = []  # Layer sizes of the network that only belongs to the policy network (low level)
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # Iterate through the shared layers and build the shared parts of the network
        for layer in net_arch:
            if isinstance(layer, int):  # Check that this is a shared layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared, layer))  # add linear of size layer
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers_h = layer["pi"]
                    policy_only_layers_l = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value networ

        last_layer_dim_pi_h = last_layer_dim_shared
        last_layer_dim_pi_l = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for pi_layer_size_h, pi_layer_size_l, vf_layer_size in zip_longest(policy_only_layers_h, policy_only_layers_l, value_only_layers):
            if pi_layer_size_h is not None:
                assert isinstance(pi_layer_size_h, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net_h.append(nn.Linear(last_layer_dim_pi_h, pi_layer_size_h))
                policy_net_h.append(activation_fn())
                last_layer_dim_pi_h = pi_layer_size_h
            if pi_layer_size_l is not None:
                assert isinstance(pi_layer_size_l, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net_l.append(nn.Linear(last_layer_dim_pi_l, pi_layer_size_l))
                policy_net_l.append(activation_fn())
                last_layer_dim_pi_l = pi_layer_size_l

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi_h = last_layer_dim_pi_h
        self.latent_dim_pi_l = last_layer_dim_pi_l
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net_h = nn.Sequential(*policy_net_h).to(device)
        self.policy_net_l = nn.Sequential(*policy_net_l).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net_h(shared_latent), self.policy_net_l(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net_h(self.shared_net(features)), self.policy_net_l(self.shared_net(features))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(self.shared_net(features))

class HybridActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space -> tuple of discrete and continuous action spaces ##
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: tuple,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        action_space_h, action_space_l = self.action_space
        # Action distribution
        self.action_dist_h = make_masked_proba_distribution(action_space_h)
        self.action_dist_l = make_proba_distribution(action_space_l, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.
        :param n_envs:
        """
        assert isinstance(self.action_dist_h, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist_l, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist_h.sample_weights(self.log_std_h, batch_size=n_envs)
        self.action_dist_l.sample_weights(self.log_std_l, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = HybridMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi_h = self.mlp_extractor.latent_dim_pi_h
        latent_dim_pi_l = self.mlp_extractor.latent_dim_pi_l

        if isinstance(self.action_dist_h, DiagGaussianDistribution):
            self.action_net_h, self.log_std_h = self.action_dist_h.proba_distribution_net(
                latent_dim=latent_dim_pi_h, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist_h, StateDependentNoiseDistribution):
            self.action_net_h, self.log_std_h = self.action_dist_h.proba_distribution_net(
                latent_dim=latent_dim_pi_h, latent_sde_dim=latent_dim_pi_h, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist_h, (MaskableDistribution, CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net_h = self.action_dist_h.proba_distribution_net(latent_dim=latent_dim_pi_h)
        else:
            raise NotImplementedError(f"Unsupported high level action distribution '{self.action_dist_h}'.")
        
        if isinstance(self.action_dist_l, DiagGaussianDistribution):
            self.action_net_l, self.log_std_l = self.action_dist_l.proba_distribution_net(
                latent_dim=latent_dim_pi_l, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist_l, StateDependentNoiseDistribution):
            self.action_net_l, self.log_std_l = self.action_dist_l.proba_distribution_net(
                latent_dim=latent_dim_pi_l, latent_sde_dim=latent_dim_pi_l, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist_l, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net_l = self.action_dist_l.proba_distribution_net(latent_dim=latent_dim_pi_l)
        else:
            raise NotImplementedError(f"Unsupported low level action distribution '{self.action_dist_l}'.")

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
                self.action_net_h: 0.01,
                self.action_net_l: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False, action_masks: Optional[np.ndarray] = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi_h, latent_pi_l, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution_h = self._get_action_dist_h_from_latent(latent_pi_h)
        distribution_l = self._get_action_dist_l_from_latent(latent_pi_l)
        if action_masks is not None:
            distribution_h.apply_masking(action_masks)
        actions_h = distribution_h.get_actions(deterministic=deterministic)
        actions_l = distribution_l.get_actions(deterministic=deterministic)
        log_prob_h = distribution_h.log_prob(actions_h)
        log_prob_l = distribution_l.log_prob(actions_l)
        action_space_h, action_space_l = self.action_space
        actions_h = actions_h.reshape((-1,) + action_space_h.shape) 
        actions_l = actions_l.reshape((-1,) + action_space_l.shape) 
        actions = (actions_h, actions_l)
        log_prob = (log_prob_h, log_prob_l)
        return actions, values, log_prob

    # Ao's code
    def _get_action_dist_h_from_latent(self, latent_pi_h: th.Tensor) -> MaskableDistribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        action_logits = self.action_net_h(latent_pi_h)
        return self.action_dist_h.proba_distribution(action_logits=action_logits)

    def _get_action_dist_l_from_latent(self, latent_pi_l: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net_l(latent_pi_l)

        if isinstance(self.action_dist_l, DiagGaussianDistribution):
            return self.action_dist_l.proba_distribution(mean_actions, self.log_std_l)
        elif isinstance(self.action_dist_l, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist_l.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist_l, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist_l.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist_l, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist_l.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist_l, StateDependentNoiseDistribution):
            return self.action_dist_l.proba_distribution(mean_actions, self.log_std_l, latent_pi_l)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, deterministic: bool = False, action_masks: Optional[np.ndarray]=None) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.
        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        # return self.get_distribution(observation).get_actions(deterministic=deterministic)
        return self.get_distribution(observation, action_masks)[0].get_actions(deterministic=deterministic), self.get_distribution(observation)[1].get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor, action_masks: Optional[np.ndarray] = None) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi_h, latent_pi_l, latent_vf = self.mlp_extractor(features)
        distribution_h = self._get_action_dist_h_from_latent(latent_pi_h)
        if action_masks is not None:
            distribution_h.apply_masking(action_masks)
        distribution_l = self._get_action_dist_l_from_latent(latent_pi_l)
        log_prob_h = distribution_h.log_prob(actions[0])
        log_prob_l = distribution_l.log_prob(actions[1])
        values = self.value_net(latent_vf)
        entropy_h = distribution_h.entropy()
        entropy_l = distribution_l.entropy()
        return values, log_prob_h, log_prob_l, entropy_h, entropy_l

    # Ao's code
    def get_distribution(self, obs: th.Tensor, action_masks: Optional[np.ndarray] = None) -> Tuple[MaskableDistribution, Distribution]:
        """
        Get the current policy distribution given the observations.
        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi_h, latent_pi_l = self.mlp_extractor.forward_actor(features)
        distribution_h = self._get_action_dist_h_from_latent(latent_pi_h)
        distribution_l = self._get_action_dist_l_from_latent(latent_pi_l)
        if action_masks is not None:
            distribution_h.apply_masking(action_masks)
        return distribution_h, distribution_l

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        :param obs:
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None, # Ao's code
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :param action_masks: Action masks to apply to the action distribution
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if episode_start is None:
        #     episode_start = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)
        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic, action_masks=action_masks)

        # Convert to numpy, and reshape to the original action shape
        actions_h = actions[0].item()
        actions_l = np.array(actions[1][0])
        # actions = actions.cpu().numpy().reshape((-1,) + self.action_space.shape)
        action_space_h, action_space_l = self.action_space
        if isinstance(action_space_h, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions_h = self.unscale_action(actions_h)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions_h = np.clip(actions_h, action_space_h.low, action_space_h.high)
                
        if isinstance(action_space_l, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions_l = self.unscale_action(actions_l)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions_l = np.clip(actions_l, action_space_l.low, action_space_l.high)

        # Remove batch dimension if needed
        # if not vectorized_env:
        #     actions = actions.squeeze(axis=0)
            
        actions = np.array([actions_h, actions_l])


        return actions, state