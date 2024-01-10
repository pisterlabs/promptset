# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
#from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, register_policy
#from stable_baselines3.common.policies import ActorCriticCnnPolicy, register_policy
from stable_baselines3.common.policies import register_policy

#MlpPolicy = ActorCriticPolicy
#CnnPolicy = ActorCriticCnnPolicy

#register_policy("MlpPolicy", ActorCriticPolicy)
#register_policy("CnnPolicy", ActorCriticCnnPolicy)

import collections
import copy
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

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
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from stable_baselines3.common.policies import BaseModel, BasePolicy

class MiniQNetwork(nn.Module):
    #def __init__(self, in_channels, num_actions): # org
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):

        super(MiniQNetwork, self).__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

        n_input_channels = observation_space.shape[0]

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        #self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1) # org
        self.conv = nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1)

        ## Final fully connected hidden layer:
        ##   the number of linear unit depends on the output of the conv
        ##   the output consist 128 rectified units
        #def size_linear_unit(size, kernel_size=3, stride=1):
        #    return (size - (kernel_size - 1) - 1) // stride + 1
        #num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        #self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
        with th.no_grad():
            n_flatten = (nn.Flatten()(self.conv(th.as_tensor(observation_space.sample()[None]).float()))).shape[1]

        self.fc_hidden = nn.Linear(in_features=n_flatten, out_features=128)

        # Output layer:
        #self.output = nn.Linear(in_features=128, out_features=num_actions) # org
        self.output = nn.Linear(in_features=128, out_features=features_dim)


    @property
    def features_dim(self) -> int:
        return self._features_dim

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    #def forward(self, x):
    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = nn.Flatten()(nn.functional.relu(self.conv(observations)))
        #print(x.shape)
        x = nn.functional.relu(self.fc_hidden(x))
        #print(x.shape)
        #return self.output(nn.functional.relu(self.fc_hidden(nn.functional.relu(self.conv(observations)).view(observations.size(0), -1))))
        #return self.output(nn.functional.relu(self.fc_hidden(nn.functional.relu(self.conv(observations)).view(observations.size(0), -1))))
        return self.output(x)
        ## Rectified output from the first conv layer
        #x = f.relu(self.conv(x))

        ## Rectified output from the final hidden layer
        #x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))

        ## Returns the output from the fully-connected linear layer
        #return self.output(x)


class ActorCriticPolicy2(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
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
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
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

        super(ActorCriticPolicy2, self).__init__(
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
            elif features_extractor_class == MiniQNetwork:
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
                "learn_features": sde_net_arch is not None,
            }

        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

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
                sde_net_arch=self.sde_net_arch,
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
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
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

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate features extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        # self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # org version
        #self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # q-value version (vf + a)
        #self.q_value_net = nn.Linear(self.mlp_extractor.latent_dim_vf + 1, 1)
        #self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, latent_dim_pi)
        '''Note: Our value_net has action_space.n outputs, which estimates the Q value'''
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, self.action_space.n)
        #self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 4)
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
                self.value_net: 1,
                #self.q_value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        OUTPUT
        :values: with action space dimension
        :log_prob: only the prob of the chosen action
        :actions: get actions from policy (if deterministic == True, get_action will return action with the largest prob)
        """
        # print("forward obs:",obs)
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # print("forward latent_pi:",latent_pi,"latent_sde",latent_sde)
        # Evaluate the values for the given observations
        # org version
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)

        #print(latent_vf.shape, th.unsqueeze(actions, 1).shape) 
        #print(th.cat((latent_vf, th.unsqueeze(actions, 1)), -1).shape) 
        #q_values = self.q_value_net(th.cat((latent_vf, th.unsqueeze(actions, 1)), -1))
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob
        #return actions, q_values, log_prob

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.
        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.
        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        # print("distribution",vars(distribution['distribution']))
        # print("distribution", distribution[distribution] )
        # print("log_prob_from_params",distribution.log_prob_from_params(distribution))
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.

        OUTPUT
        :values: with action space dimension (All Q(s,a) of given state s)
        :log_prob: only the prob of the chosen action
        :entropy: the entropy of the distribution of the state s
        """
        # print("evaluate_actions obs",obs)
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # print("evaluate_actions latent_pi:",latent_pi,"latent_sde",latent_sde)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        #q_values = self.q_value_net(th.cat((latent_vf, th.unsqueeze(actions, 1)), -1))
        return values, log_prob, distribution.entropy()
        #return q_values, log_prob, distribution.entropy()

class ActorCriticCnnPolicy2(ActorCriticPolicy2):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(ActorCriticCnnPolicy2, self).__init__(
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

    #def _build(self, lr_schedule: Schedule) -> None:
    #    """
    #    Create the networks and the optimizer.
    #    :param lr_schedule: Learning rate schedule
    #        lr_schedule(1) is the initial learning rate
    #    """
    #    self._build_mlp_extractor()

    #    latent_dim_pi = self.mlp_extractor.latent_dim_pi

    #    # Separate features extractor for gSDE
    #    if self.sde_net_arch is not None:
    #        self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
    #            self.features_dim, self.sde_net_arch, self.activation_fn
    #        )

    #    if isinstance(self.action_dist, DiagGaussianDistribution):
    #        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
    #            latent_dim=latent_dim_pi, log_std_init=self.log_std_init
    #        )
    #    elif isinstance(self.action_dist, StateDependentNoiseDistribution):
    #        latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
    #        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
    #            latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
    #        )
    #    elif isinstance(self.action_dist, CategoricalDistribution):
    #        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
    #    elif isinstance(self.action_dist, MultiCategoricalDistribution):
    #        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
    #    elif isinstance(self.action_dist, BernoulliDistribution):
    #        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
    #    else:
    #        raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

    #    #self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
    #    self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, latent_dim_pi)
    #    # Init weights: use orthogonal initialization
    #    # with small initial weight for the output
    #    if self.ortho_init:
    #        # TODO: check for features_extractor
    #        # Values from stable-baselines.
    #        # features_extractor/mlp values are
    #        # originally from openai/baselines (default gains/init_scales).
    #        module_gains = {
    #            self.features_extractor: np.sqrt(2),
    #            self.mlp_extractor: np.sqrt(2),
    #            self.action_net: 0.01,
    #            self.value_net: 1,
    #        }
    #        for module, gain in module_gains.items():
    #            module.apply(partial(self.init_weights, gain=gain))

    #    # Setup optimizer with initial learning rate
    #    self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

MlpPolicy = ActorCriticPolicy2
register_policy("MlpPolicy", ActorCriticPolicy2)
CnnPolicy = ActorCriticCnnPolicy2
#register_policy("CnnPolicy", CnnPolicy)
register_policy("CnnPolicy", ActorCriticCnnPolicy2)
