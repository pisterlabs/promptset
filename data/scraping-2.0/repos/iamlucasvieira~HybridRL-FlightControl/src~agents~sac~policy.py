"""Create policy for SAC algorithm."""
from typing import List, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.distributions import Normal
from torch.nn.functional import softplus

from agents import BasePolicy
from helpers.torch_helpers import BaseNetwork, mlp


class CriticNetwork(BaseNetwork):
    """Creates the critic neural network."""

    def __init__(
        self, observation_space: spaces.Box, action_space: spaces.Box, **kwargs
    ):
        """Initialize critic network.

        args:
            beta: Learning rate.
            input_dims: Input dimensions.
            n_actions: Number of actions.
            fc1_dims: Number of neurons in the first layer.
            fc2_dims_: Number of neurons in the second layer.
        """
        # Input layer of critic's neural network in SAC uses state-action pairs
        super().__init__(observation_space, action_space, **kwargs)

    def _build_network(self) -> nn.Sequential:
        """Build network."""
        ff = mlp(
            [self.observation_dim + self.action_dim] + self.hidden_layers + [1],
            activation=nn.ReLU,
            layer_norm=False,
        )
        return ff

    def forward(self, state, action):
        """Forward pass of the critic's neural network."""
        q = self.ff(th.cat([state, action], dim=-1))
        return th.squeeze(q, -1)


class ActorNetwork(BaseNetwork):
    """Actor network in SAC."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        sigma_min: float = -30,
        sigma_max: float = 2,
        **kwargs
    ):
        """Initialize actor network.

        args:
            observation_space: Observation space.
            action_space: Action space.
            sigma_min: Minimum value of the standard deviation.
            sigma_max: Maximum value of the standard deviation.
        """
        super().__init__(observation_space, action_space, **kwargs)
        self.mu = nn.Linear(self.hidden_layers[-1], self.action_dim)
        self.log_sigma = nn.Linear(self.hidden_layers[-1], self.action_dim)
        self.action_max = float(action_space.high[0])

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.to(self.device)

    def _build_network(self) -> nn.Sequential:
        """Build network."""
        ff = mlp(
            [self.observation_dim] + self.hidden_layers,
            activation=nn.ReLU,
            output_activation=nn.ReLU,
            layer_norm=False,
        )
        return ff

    def output_layer(self, net_output, with_log_prob=True, deterministic=False):
        """The output layer of the SAC agent."""
        mu = self.mu(net_output)
        log_sigma = th.clamp(
            self.log_sigma(net_output), min=self.sigma_min, max=self.sigma_max
        )
        sigma = th.exp(log_sigma)

        action_distribution = Normal(mu, sigma)

        action = action_distribution.rsample() if not deterministic else mu
        action.to(self.device)

        if with_log_prob:  # From OpenAi Spinning Up
            log_prob = action_distribution.log_prob(action) - 2 * (
                np.log(2) - action - softplus(-2 * action)
            )
            log_prob = log_prob.sum(axis=-1)
        else:
            log_prob = None

        action = th.tanh(action)  # * self.action_max

        return action, log_prob

    def forward(self, state: th.Tensor, with_log_prob=True, deterministic=False):
        """Forward pass in the actor network.

        args:
            state: State.
            with_log_prob: Whether to return the log probability.
        """
        net_output = self.ff(state)
        action, log_prob = self.output_layer(
            net_output, with_log_prob=with_log_prob, deterministic=deterministic
        )

        return action, log_prob


class SACPolicy(BasePolicy):
    """Policy for SAC algorithm."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        learning_rate: float = 3e-4,
        hidden_layers: List[int] = None,
        device: Optional[str] = None,
    ):
        """Initialize policy.

        args:
            observation_space: Observation space.
            action_space: Action space.
            learning_rate: Learning rate.
            hidden_layers: Number of hidden layers.
            save_path: Path to save the policy.

        """
        if hidden_layers is None:
            hidden_layers = [64, 64]

        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        super().__init__(observation_space, action_space, device=device)

    def _setup_policy(self):
        """Setup policy."""
        observation_space = self.observation_space
        action_space = self.action_space
        learning_rate = self.learning_rate
        hidden_layers = self.hidden_layers

        self.actor = ActorNetwork(
            observation_space,
            action_space,
            learning_rate=learning_rate,
            hidden_layers=hidden_layers,
            device=self.device,
        )

        self.critic_1 = CriticNetwork(
            observation_space,
            action_space,
            learning_rate=learning_rate,
            hidden_layers=hidden_layers,
            device=self.device,
        )

        self.critic_2 = CriticNetwork(
            observation_space,
            action_space,
            learning_rate=learning_rate,
            hidden_layers=hidden_layers,
            device=self.device,
        )

    def get_action(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """Get action from the policy.

        args:
            state: Current state.

        returns:
            action: Action to take.
        """
        with th.no_grad():
            state = th.tensor(state, dtype=th.float32, device=self.actor.device)
            action, _ = self.actor(state, **kwargs)
        return action.cpu().numpy()

    def _predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        """Predict action."""
        return self.get_action(observation, deterministic=deterministic)
