from typing import Optional
from typing import Sequence
from typing import List

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from portfolio_management.soft_actor_critic.utilities import weight_initialization
from portfolio_management.soft_actor_critic.utilities import get_multilayer_perceptron

from portfolio_management.soft_actor_critic.evaluators import Evaluator


class StochasticPolicy(nn.Module):
    def __init__(
            self,
            evaluator: Evaluator,
            num_actions: int,
            hidden_units: Optional[Sequence[int]] = None,
            action_space=None,
            epsilon: float = 10e-6,
            log_sigma_max: float = 2,
            log_sigma_min: float = -20
    ):
        super(StochasticPolicy, self).__init__()

        if hidden_units is None:
            hidden_units = [256, 256]

        self.evaluator = deepcopy(evaluator)

        self.input_dims = evaluator.num_outputs
        self.num_actions = num_actions
        self.hidden_units = list(hidden_units)

        self.epsilon = epsilon
        self.log_sigma_max = log_sigma_max
        self.log_sigma_min = log_sigma_min

        units = [self.input_dims] + list(hidden_units)
        self.multilayer_perceptron = get_multilayer_perceptron(units, keep_last_relu=True)
        self.mean_linear = nn.Linear(units[-1], num_actions)
        self.log_std_linear = nn.Linear(units[-1], num_actions)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

        self.apply(weight_initialization)

    def forward(self, tensor_list: List[torch.Tensor]):  # todo maybe merge forward and evaluate + maybe add the "act" from openai
        x_evaluated = self.evaluator(*tensor_list)
        x = self.multilayer_perceptron(x_evaluated)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std_clamped = torch.clamp(log_std, min=self.log_sigma_min, max=self.log_sigma_max)
        std = torch.exp(log_std_clamped)
        return mean, std

    def evaluate(
            self,
            tensor_list: List[torch.Tensor],
            deterministic: bool = False,
            with_log_probability: bool = True
    ):
        mean, std = self.forward(tensor_list)
        distribution = Normal(mean, std)
        sample = distribution.rsample()

        if deterministic:
            action = mean
        else:
            action = torch.tanh(sample)  # todo when sac working, multiply by action_scale and add action_bias

        if with_log_probability:
            # Implementation that I originally implemented
            # the "_" are only here for now to debug the values and the shapes
            # log_probability_ = distribution.log_prob(sample) - torch.log((1 - action.pow(2)) + self.epsilon)
            # log_probability = log_probability_.sum(1, keepdim=True)

            # OPENAI Implementation
            # https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/core.py#L59
            log_probability_ = distribution.log_prob(sample).sum(axis=-1, keepdim=True)
            log_probability__ = (2 * (np.log(2) - sample - F.softplus(-2 * sample))).sum(axis=1).unsqueeze(1)
            log_probability = log_probability_ - log_probability__
        else:
            log_probability = None

        return action, log_probability

    def act(
            self,
            tensor_list: List[torch.Tensor],
            deterministic=False
    ) -> np.array:
        with torch.no_grad():
            action, _ = self.evaluate(
                tensor_list,
                deterministic=deterministic,
                with_log_probability=False
            )
            return action.squeeze(0).cpu().numpy() # todo maybe issue dimension DDD
