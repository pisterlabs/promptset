import numpy as np
import collections
import copy
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn

from tools.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from tools.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
    RNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor

class ActorCriticPolicy(nn.Module):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param hidden_dim: the hidden state dimension of RNN
    :param use_sde: Whether to use State Dependent Exploration or not
    :param use_last_action: Whether to use the last action as input 
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
        agent_num: int = 4,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        hidden_dim: int = 16,
        use_sde: bool = False,
        use_last_action: bool = True,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(ActorCriticPolicy, self).__init__()
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        self.observation_space = observation_space
        self.action_space = action_space
        self.agent_num = agent_num
        self.normalize_images = normalize_images

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None  # type: Optional[th.optim.Optimizer]

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.features_extractor = nn.Flatten()
        # self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        # self.features_dim = self.features_extractor.features_dim
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
        self.use_last_action = use_last_action
        self.dist_kwargs = dist_kwargs
        self.squash_output = squash_output
        self.hidden_dim = hidden_dim
        # self.features_dim = self.get_mlp_input_shape()
        self.actor_inpu_dim = self.get_actor_input_shape() #这个应该得等于上面的mlp input shape
        self.critic_input_dim = self.get_critic_input_shape()
        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)
        # Buuld RNN 
        self.RNN =RNN(input_shape=1, hidden_dim=self.hidden_dim)  # only the close price as input, so the shape is 1


        self._build(lr_schedule)

    # def _get_constructor_parameters(self) -> Dict[str, Any]:
    #     data = super()._get_constructor_parameters()

    #     default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

    #     data.update(
    #         dict(
    #             net_arch=self.net_arch,
    #             activation_fn=self.activation_fn,
    #             use_sde=self.use_sde,
    #             log_std_init=self.log_std_init,
    #             squash_output=default_none_kwargs["squash_output"],
    #             full_std=default_none_kwargs["full_std"],
    #             sde_net_arch=self.sde_net_arch,
    #             use_expln=default_none_kwargs["use_expln"],
    #             lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
    #             ortho_init=self.ortho_init,
    #             optimizer_class=self.optimizer_class,
    #             optimizer_kwargs=self.optimizer_kwargs,
    #             features_extractor_class=self.features_extractor_class,
    #             features_extractor_kwargs=self.features_extractor_kwargs,
    #         )
    #     )
    #     return data

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
            actor_feature_dim=self.actor_inpu_dim,
            critic_feature_dim=self.critic_input_dim,
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
        # if self.sde_net_arch is not None:
        #     self.sde_features_extractor, latent_sde_dim = self.create_sde_features_extractor(
        #         self.features_dim, self.sde_net_arch, self.activation_fn
        #     )

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
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, last_action: th.Tensor, agend_id: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_sde = self._get_latent(obs, last_action, agend_id)   #   obs.shape = [1, 30, 252]
        # Evaluate the values for the given observations
        # values = self.value_net(latent_vf)
        distribution, action_mean = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions()
        log_prob = distribution.log_prob(actions)
        return actions, log_prob, action_mean

    def _get_latent(self, obs: th.Tensor, last_action: th.Tensor, agend_id: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation.shape=(agent_num, stock_num, lookback)
        :param last_action: action execute at last time, shape=(agent_num, stock_num)
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        assert obs.shape[0] == last_action.shape[0], "buffer size or agent num inconsistent"

        batch_multiply_agent = obs.shape[0]
        stock_num = obs.shape[1]
        lookback = obs.shape[2]
        # obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images) # NO WORK!!!
        rnn_hidden = self.RNN(obs.reshape(batch_multiply_agent*stock_num, lookback))   # rnn_hidden shape should be (stock_dim, lookback, hidden_dim)
        rnn_hidden = rnn_hidden[:,-1,:].view(batch_multiply_agent, stock_num, self.hidden_dim)
        features = self.features_extractor(rnn_hidden)   # only use the last time step hidden state, features.shape = [1, 1920] flatten 
        features = th.cat([features, agend_id.cuda()],1)
        if self.use_last_action:
            features = th.cat([features, last_action.cuda()],1)   # concat the rnn_latent and last_action
        latent_pi = self.mlp_extractor(features, None)   # laten_pi.shape = [1, 64], None in second placeholder means the actor mlp extractor
        # Features for sde
        latent_sde = latent_pi
        # if self.sde_features_extractor is not None:
        #     latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_sde

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution, mean_action
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std), mean_actions
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions), mean_actions
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions), mean_actions
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions), mean_actions
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde), mean_actions
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, last_action: th.Tensor, agent_oh: th.Tensor) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, latent_sde = self._get_latent(observation, last_action, agent_oh)
        distribution, action_mean = self._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution.get_actions()

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        last_action:  Union[np.ndarray, Dict[str, np.ndarray]],
        agent_oh:  Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action and state from an observation (and optional state).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if mask is None:
        #     mask = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # observation, vectorized_env = self.obs_to_tensor(observation)
        observation = obs_as_tensor(observation, self.device)

        with th.no_grad():
            actions = self._predict(observation, last_action, agent_oh)
        # Convert to numpy
        agent_actions = actions.cpu().numpy()
        joint_action = np.mean(agent_actions, axis=0)

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(joint_action)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(joint_action, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        # if not vectorized_env:
        #     actions = actions[0]
        return actions.reshape(1,-1), agent_actions

    def evaluate_actions(self, state: th.Tensor, obs: th.Tensor, actions: th.Tensor, last_action: th.Tensor, agent_oh: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param state: (1, lookback, stock_num)
        :param obs: (agent_num, lookback, stock_num)
        :param actions: (agent_num, stock_num)
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # last_actions = th.cat((th.zeros((1, actions.shape[1])).cuda(), actions[1:]), axis=0)
        latent_pi, latent_sde = self._get_latent(obs.permute(0,2,1), last_action, agent_oh)   #   obs.permute(0,2,1).shape = [agent_num, 30, 252]
        # Evaluate the values for the given observations
        distribution, action_mean = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        # actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        critic_input_latent, critic_cf_input_latent = self.get_critic_input(actions, obs.permute(0,2,1), state[0].t(), action_mean)
        values = self.value_net(critic_input_latent)  # values.shape=[agent_num, 1]
        values_baseline = self.value_net(critic_cf_input_latent)
        # latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        # log_prob = distribution.log_prob(actions)
        # values = self.value_net(latent_vf)
        return values, values_baseline, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        latent_pi, _, latent_sde = self._get_latent(obs)
        distribution, action_mean = self._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        _, latent_vf, _ = self._get_latent(obs)
        return self.value_net(latent_vf)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs)

    def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[th.Tensor, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = obs_as_tensor(observation, self.device)
        return observation, vectorized_env

    def get_actor_input_shape(self) -> Tuple[int]:
        input_shape = 0
        input_shape += self.agent_num
        input_shape += self.hidden_dim * self.action_space.shape[0]
        if self.use_last_action:
            input_shape += self.action_space.shape[0]

        return input_shape

    def get_critic_input_shape(self) -> Tuple[int]:
        # the critic input shape equal to observation shape + state shape + action shape
        input_shape = 0
        input_shape += self.agent_num
        # all agent action
        input_shape += self.action_space.shape[0] * self.agent_num    # stock_num * agent_num
        input_shape += self.hidden_dim * self.action_space.shape[0] * 2  # state space + obs space

        return input_shape

    def get_critic_input(self, actions: th.Tensor, obs: th.Tensor, state: th.Tensor, guassian_mean: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        '''
        concat the action, state, obs into a vector and compress the dim by using mlp
        Input:  actions.shape = (agent_num=4, action_dim=30)
                obs.shape = (agent_num=4, obs_dim=(30,252))
                state.shape =(30, 252)
                guassian_mean.shape=(agent_num=4, action_dim=30)
        Output: critic_input_latent.shape = (agent_num, hidden_dim=64)
        '''
        critic_input = []
        critic_cf_input = []         # the counterfactual baseline input, replace the agents' actions with Guassian mean
        assert actions.shape[0] == obs.shape[0], "inconsistent agent num"
        agent_num = actions.shape[0]
        stock_num = obs.shape[1]
        actions = actions.view((1, -1)).repeat(agent_num, 1)  # reshape and repeat to (agent_num, 120), contains all agents actions
        guassian_mean = guassian_mean.view((1,-1)).repeat(agent_num, 1).cuda()
        # replace the agents' action with Guassian mean
        action_mask = (1 - th.eye(self.agent_num)) # use a mask to ignore the action of agent itself, but no the other agents
        action_mask = action_mask.view(-1, 1).repeat(1, stock_num).view(agent_num, -1).cuda()   #shape=(agent_num, 120)
        guassian_mask = th.eye(self.agent_num)
        guassian_mask = guassian_mask.view(-1, 1).repeat(1, stock_num).view(agent_num, -1).cuda()
        # critic_input.append(actions*action_mask) 
        critic_input.append(actions)
        critic_cf_input.append(actions*action_mask + guassian_mean*guassian_mask)

        # add the agent id 
        critic_input.append(th.eye(agent_num).cuda())
        critic_cf_input.append(th.eye(agent_num).cuda())

        # use GRU to extract state representation
        state_hidden = self.RNN(state)  # state_hidden.shape=(stock_num, hidden_size)
        flat_state_hidden = self.features_extractor(state_hidden[:,-1,:].unsqueeze(0))
        latent_state = flat_state_hidden.view(1, -1).repeat(agent_num, 1)
        critic_input.append(latent_state)  #  shape=(agent_num, 1920)
        critic_cf_input.append(latent_state)

        # use GRU to extract obs representation
        _obs = obs.reshape((agent_num*stock_num, -1))
        obs_hidden = self.RNN(_obs)    #obs_hidden.shape=(agent_num*stock_num, 252, self.hidden_dim)
        obs_hidden = obs_hidden[:,-1,:].view((agent_num, stock_num, -1))
        latent_obs = obs_hidden.view((agent_num, -1))
        critic_input.append(latent_obs)  #(agent_num, 1920)
        critic_cf_input.append(latent_obs)
        critic_input = th.cat([x.view(agent_num, -1) for x in critic_input], dim=1) #(agent_num, 1920+1920+120)
        critic_cf_input = th.cat([x.view(agent_num, -1) for x in critic_cf_input], dim=1)
        
        critic_input_latent = self.mlp_extractor(None, critic_input) #(agent_num, 64)
        critic_cf_input_latent = self.mlp_extractor(None, critic_cf_input)

        return critic_input_latent, critic_cf_input_latent

    # def create_sde_features_extractor(self, features_dim, sde_net_arch, activation_fn) -> None:

    #     return