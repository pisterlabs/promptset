# This file is here just to define MlpPolicy/CnnPolicy
# that work for A2C
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, register_policy, BasePolicy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from functools import partial
import gym
import torch as th
from torch import nn
import copy
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, create_sde_features_extractor, register_policy
from stable_baselines3.common.preprocessing import get_action_dim

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
    SoftmaxCategorical
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_action_dim, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor, NatureCNN, create_mlp
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper


class OffPACPolicy(BasePolicy):
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
        activation_fn: Type[nn.Module] = nn.ReLU,
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
        share: bool = True,
        uniform_sampling=False
    ):
        self.uniform_sampling = uniform_sampling
        self.share = share
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(OffPACPolicy, self).__init__(
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
            if features_extractor_class == FlattenExtractor:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]
            else:
                net_arch = []

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
        self.q_net, self.q_net_target, self.behav_net = None, None, None
        self.value_net = None
        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, softmax=True, dist_kwargs=dist_kwargs)

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
        if self.share:
            self.v_mlp_extractor = MlpExtractor(
                self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device
            )
            self.v_mlp_extractor_target = self.v_mlp_extractor
            self.a_mlp_extractor = self.v_mlp_extractor
            self.a_mlp_extractor_target = self.v_mlp_extractor
        else:
            self.v_mlp_extractor = MlpExtractor(
                self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device
            )
            self.v_mlp_extractor_target = copy.deepcopy(self.v_mlp_extractor)
            self.a_mlp_extractor = MlpExtractor(
                self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device
            )
            self.a_mlp_extractor_target = copy.deepcopy(self.a_mlp_extractor)



    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.a_mlp_extractor.latent_dim_pi

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
        elif isinstance(self.action_dist, SoftmaxCategorical):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.q_net = nn.Linear(self.v_mlp_extractor.latent_dim_vf, self.action_space.n)
        self.q_net_target = nn.Linear(self.v_mlp_extractor.latent_dim_vf, self.action_space.n)
        self.value_net = nn.Linear(self.v_mlp_extractor.latent_dim_vf, 1)

        # self.behav_net = nn.Linear(self.mlp_extractor.latent_dim_vf, self.action_space.n)
        
        # self.behav_net.load_state_dict(self.q_net_target.state_dict())

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.v_mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.q_net: 1,
                self.value_net: 1
            }

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.behav_net = copy.deepcopy(self.action_net)
        # self.behav_net = None
        # self.behav_net = self.action_net
        '''
        for name, param in self.named_parameters():
            if param.requires_grad:
                print (name, param.data[0])
        '''

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        pass
        # latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # # Evaluate the values for the given observations
        # values = self.q_net(latent_vf)
        # distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        # actions = distribution.get_actions(deterministic=deterministic)
        # log_prob = distribution.log_prob(actions)
        # return actions, values, log_prob

    def _get_latent(self, obs: th.Tensor, use_target_v: bool = False, use_behav:bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share:
            latent_pi, latent_vf = self.a_mlp_extractor(features)
        else:
            if use_target_v:
                _, latent_vf = self.v_mlp_extractor_target(features)
            else:
                _, latent_vf = self.v_mlp_extractor(features)
            
            if use_behav:
                latent_pi, _ = self.a_mlp_extractor_target(features)
            else:
                latent_pi, _ = self.a_mlp_extractor(features)
                

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None, use_behav: bool = False) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        if use_behav:
            action_net = self.behav_net
        else:
            action_net = self.action_net
        action_net = self.action_net
        mean_actions = action_net(latent_pi)


        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, SoftmaxCategorical):
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

    def _predict(self, observation: th.Tensor, deterministic: bool = False, use_behav: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        if use_behav:
            action_net = self.behav_net
        else:
            action_net = self.action_net

        latent_pi, _, latent_sde = self._get_latent(observation, use_behav=use_behav)
        if self.uniform_sampling and use_behav:
            distribution = self.action_dist.proba_distribution(action_logits=th.zeros_like(action_net(latent_pi)))
        else:
            distribution = self._get_action_dist_from_latent(latent_pi, latent_sde, use_behav)

        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor, use_target_v: bool = True, use_behav: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # latent_pi, latent_vf, latent_sde = self._get_latent(obs, use_target)
        latent_pi, latent_vf, latent_sde = self._get_latent(obs, use_target_v=use_target_v, use_behav=use_behav)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde, use_behav)
        # print(th.exp(distribution.log_prob(actions.squeeze())))
        # for i in range(obs.size(0)):
            # latent_pi, latent_vf, latent_sde = self._get_latent(obs[i])
            # distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
            # print(th.exp(distribution.log_prob(actions[i])))
        # print()
        # exit()

        log_prob = distribution.log_prob(actions.squeeze())
        
        q_net = self.q_net_target if use_target_v else self.q_net
        Q_values = th.gather(q_net(latent_vf), dim=1, index=actions.detach().long())
        # print(self.q_net(latent_vf))
        # print(actions)
        # print(Q_values)
        # exit()
        return Q_values, log_prob, distribution.entropy()

    def get_action_log_probs(self, obs, actions, use_behav=False):
        assert obs.size(0) == actions.size(0)

        latent_pi, latent_vf, latent_sde = self._get_latent(obs, use_behav=use_behav)

        if self.uniform_sampling and use_behav:
            distribution = self.action_dist.proba_distribution(action_logits=th.zeros_like(self.action_net(latent_pi)))
        else:
            distribution = self._get_action_dist_from_latent(latent_pi, latent_sde, use_behav)

        log_probs_grid = distribution.log_prob(actions) # row 0 is (s0, a0), (s1, a0)...(sn, a0)

        # print(distribution.log_prob(th.tensor([0]).to(self.device)))
        # print(distribution.log_prob(th.tensor([1]).to(self.device)))
        # print("grid")
        # print(log_probs_grid.size())
        
        assert log_probs_grid.size() == (obs.size(0), actions.size(0))
        log_probs = th.gather(log_probs_grid, dim=1, index=th.tensor([[i] for i in range(actions.size(0))]).to(self.device))

        return log_probs



    def compute_value(self, obs: th.Tensor, use_target_v: bool, use_behav: bool = False):
        """
        Compute V(s)

        :return: V(s)
        """
        '''
        q_net = self.q_net_target if use_target_v else self.q_net
        print(obs[0])
        for i in range(obs.size(0)):
            latent_pi, latent_vf, latent_sde = self._get_latent(obs[i].unsqueeze(0), use_target_v)
            Q = q_net(latent_vf)
            print(Q)
            distribution = self._get_action_dist_from_latent(latent_pi, latent_sde, use_behav)
            actions = th.tensor([[i] for i in range(self.action_space.n)]).to(self.device)
            log_prob = distribution.log_prob(actions)
            prob = th.exp(log_prob).permute(1, 0).detach()
            print(prob)
        '''

        q_net = self.q_net_target if use_target_v else self.q_net
        latent_pi, latent_vf, latent_sde = self._get_latent(obs, use_target_v)

        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde, use_behav)
        
        Q = q_net(latent_vf)

        actions = th.tensor([[i] for i in range(self.action_space.n)]).to(self.device)
        log_prob = distribution.log_prob(actions)
        prob = th.exp(log_prob).permute(1, 0).detach()


        try:
            assert Q.size() == prob.size()
        except AssertionError as e:
            print("Q.size(): ", Q.size())
            print("prob.size(): ", prob.size())
        return th.sum(Q * prob, axis=1)

    def get_policy_latent(self, obs: th.Tensor, use_behav: bool = False):
        '''
        return theta used to compute Q(s,a) with softmax
        '''
        latent_pi, latent_vf, latent_sde = self._get_latent(obs, use_behav)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde, use_behav)
        action_net = self.behav_net if use_behav else self.action_net
        return action_net(latent_pi), distribution.distribution




class OffPACCnnPolicy(OffPACPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
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
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        share: bool = True,
        uniform_sampling=False
    ):

        super(OffPACCnnPolicy, self).__init__(
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
            share = share,
            uniform_sampling=uniform_sampling
        )



MlpPolicy = OffPACPolicy
CnnPolicy = OffPACCnnPolicy

register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)
