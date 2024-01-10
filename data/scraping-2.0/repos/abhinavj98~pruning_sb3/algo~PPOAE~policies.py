from stable_baselines3.common.policies import BasePolicy
from typing import Any, Dict, List, Optional, Type, Tuple
from functools import partial
import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.type_aliases import Schedule
import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
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


class ActorCriticWithAePolicy(BasePolicy):
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
        actor_class: Type[nn.Module],
        critic_class: Type[nn.Module],
        critic_kwargs: Optional[Dict[str, Any]] = None,
        actor_kwargs: Optional[Dict[str, Any]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.1,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_schedule_ae: Schedule = 0.0001,
        lr_schedule_logstd: Schedule = 0.0001,
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

  
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        
        self.actor_class = actor_class
        self.critic_class = critic_class
        self.actor_kwargs = actor_kwargs
        self.critic_kwargs = critic_kwargs
        self.action_space = action_space
        self.use_sde = use_sde
      

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        self.dist_kwargs = dist_kwargs
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
        self._build(lr_schedule, lr_schedule_ae, lr_schedule_logstd)
        

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
    def _build_actor_critic(self) -> None:
        self.actor = self.actor_class(**self.actor_kwargs).to(self.device)
        self.critic = self.critic_class(**self.critic_kwargs).to(self.device)
        
    @staticmethod    
    def init_kaiming(m):
        if type(m) == nn.Conv2d or type(m)==nn.Linear or type(m)==nn.ConvTranspose2d:
            th.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            print("nbas")
            nn.init.orthogonal_(module.weight, gain=gain)
            print("aa")
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def _build(self, lr_schedule: Schedule, lr_schedule_ae: Schedule, lr_schedule_logstd) -> None:
        """
        Create the networks and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        #Make Actor Critic using actro critic class and kwargs, update get constructor parameters as welll
        self.features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim
        self._build_actor_critic()
        self.latent_dim_pi = self.actor.output_dim
        self.latent_dim_vf = self.critic.output_dim
        self.action_dist = SquashedDiagGaussianDistribution(get_action_dim(self.action_space))#make_proba_distribution(self.action_space, use_sde=False, dist_kwargs=self.dist_kwargs)
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=self.latent_dim_pi, log_std_init=self.log_std_init
            )
    
        self.value_net = nn.Linear(self.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
             # TODO: check for features_extractor
             # Values from stable-baselines.
             # features_extractor/mlp values are
             # originally from openai/baselines (default gains/init_scales).
             module_gains = {
                 self.features_extractor: np.sqrt(2),
                 self.actor: np.sqrt(2),
                 self.critic: np.sqrt(2),
                 self.action_net: 0.1,
                 self.value_net: 1,
             }
             for module, gain in module_gains.items():
                 print("ortho")
                 module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class([*self.actor.parameters(), *self.critic.parameters(), *self.value_net.parameters(), *self.action_net.parameters()], lr=lr_schedule(1), **self.optimizer_kwargs)
        self.optimizer_ae = self.optimizer_class(self.features_extractor.parameters(), lr=lr_schedule_ae(1), **self.optimizer_kwargs)
        self.optimizer_logstd = self.optimizer_class([self.log_std], lr=lr_schedule_logstd(1), **self.optimizer_kwargs)
        #print all nets in optimizer
        # print(self.optimizer, self.optimizer_ae)
    
    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def make_state_from_obs(self, obs):
        depth_features = self.extract_features(obs['depth'])
        #TODO: Normalize inputs
        robot_features = th.cat([obs['achieved_goal'], obs['desired_goal'], obs['joint_angles'], obs['prev_action']],  dim = 1)
        return depth_features, robot_features
        
    def forward(self, obs: Dict, deterministic: bool = False, *args, **kwargs) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
    
        # Evaluate the values for the given observations
      
        depth_features, robot_features = self.make_state_from_obs(obs)
        # print(depth_features[0].shape, robot_features.shape)
        latent_pi = self.actor(depth_features[0], robot_features)
        latent_vf = self.critic(depth_features[0].detach(), robot_features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob


    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        #preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        features = self.features_extractor(obs)
        return features
        
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: Latent code for the actor
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
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.
        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        with th.no_grad():
            depth_features, robot_features = self.make_state_from_obs(observation)
            latent_pi = self.actor(depth_features[0], robot_features)
            # latent_vf = self.critic(features[0].detach(), state)
            # values = self.value_net(latent_vf)
            distribution = self._get_action_dist_from_latent(latent_pi)
            #actions = distribution.get_actions(deterministic=deterministic)
            # log_prob = distribution.log_prob(actions)
            # return self.get_distribution(observation).get_actions(deterministic=deterministic)
            return distribution.get_actions(deterministic = deterministic)
    


    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        depth_features, robot_features = self.make_state_from_obs(obs)
        latent_pi = self.actor(depth_features[0], robot_features)
        latent_vf = self.critic(depth_features[0].detach(), robot_features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        #actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return values, depth_features, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.
        :param obs:
        :return: the action distribution.
        """
        depth_features, robot_features = self.make_state_from_obs(obs)
        latent_pi = self.actor(depth_features[0], robot_features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        :param obs:
        :return: the estimated values.
        """
        with th.no_grad():
            depth_features, robot_features = self.make_state_from_obs(obs)
            latent_vf = self.critic(depth_features[0], robot_features.float())
            return self.value_net(latent_vf)


    def set_training_mode(self, mode: bool) -> None:
            """
            Put the policy in either training or evaluation mode.
            This affects certain modules, such as batch normalisation and dropout.
            :param mode: if true, set to training mode, else set to evaluation mode
            """
            self.actor.train(mode)
            self.critic.train(mode)
            self.features_extractor.train(mode)
            self.training = mode
