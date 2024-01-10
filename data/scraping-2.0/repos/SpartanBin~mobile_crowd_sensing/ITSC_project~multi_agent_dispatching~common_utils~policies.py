from functools import partial
from typing import Tuple, Union
import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class Conv1dExtractor(nn.Module):

    def __init__(self, conv_params: list, add_BN: bool, output_dim: list, share_params: bool):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :return:
        '''
        super(Conv1dExtractor, self).__init__()

        policy_conv_net = []
        value_conv_net = []
        if add_BN:
            policy_conv_net.append(nn.BatchNorm1d(conv_params[0]['in_channels']))
            value_conv_net.append(nn.BatchNorm1d(conv_params[0]['in_channels']))
        for layer_params in conv_params:
            policy_conv_net.append(nn.Conv1d(**layer_params))
            if add_BN:
                policy_conv_net.append(nn.BatchNorm1d(layer_params['out_channels']))
            policy_conv_net.append(nn.ELU())
            value_conv_net.append(nn.Conv1d(**layer_params))
            if add_BN:
                value_conv_net.append(nn.BatchNorm1d(layer_params['out_channels']))
            value_conv_net.append(nn.ELU())
        self.policy_conv_net = nn.Sequential(*policy_conv_net)
        self.value_conv_net = nn.Sequential(*value_conv_net)

        policy_output_net = []
        value_output_net = []
        for i in range(len(output_dim)):
            if i > 0:
                policy_output_net.append(nn.Linear(output_dim[i - 1], output_dim[i]))
                policy_output_net.append(nn.Tanh())
                value_output_net.append(nn.Linear(output_dim[i - 1], output_dim[i]))
                value_output_net.append(nn.Tanh())
        self.policy_output_net = nn.Sequential(*policy_output_net)
        self.value_output_net = nn.Sequential(*value_output_net)

        self.share_params = share_params
        if self.share_params:
            del self.value_conv_net
            del self.value_output_net

    def forward(self, features: torch.Tensor):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        policy_output = self.policy_conv_net(features).flatten(start_dim=1, end_dim=-1)
        policy_output = self.policy_output_net(policy_output)
        value_output = None
        if not self.share_params:
            value_output = self.value_conv_net(features).flatten(start_dim=1, end_dim=-1)
            value_output = self.value_output_net(value_output)
        return policy_output, value_output


class CategoricalDistribution():
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: torch.Tensor) -> "CategoricalDistribution":
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def all_probs(self):
        return self.distribution.probs

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)

    def get_actions(self) -> torch.Tensor:
        """
        Return actions according to the probability distribution.
        :return:
        """
        return self.sample()

    def actions_from_params(self, action_logits: torch.Tensor) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions()

    def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class ActorCriticPolicy(nn.Module):

    def __init__(
        self,
        ortho_init: bool,
        conv_params: list,
        add_BN: bool,
        output_dim: list,
        share_params: bool,
        action_dim: int,
        learning_rate: Union[int, float],
    ):
        '''
        :param ortho_init:
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param share_params:
        :param action_dim:
        :param learning_rate:
        :return:
        '''
        super(ActorCriticPolicy, self).__init__()

        self.ortho_init = ortho_init

        self.extractor = Conv1dExtractor(
            conv_params=conv_params,
            add_BN=add_BN,
            output_dim=output_dim,
            share_params=share_params
        )
        self.value_net = nn.Linear(output_dim[-1], 1)
        # Action distribution
        self.action_distribution = CategoricalDistribution(action_dim=action_dim)
        self.action_net = self.action_distribution.proba_distribution_net(latent_dim=output_dim[-1])

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def output_action_distribution(self, latent_pi: torch.Tensor):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        return self.action_distribution.proba_distribution(action_logits=mean_actions)

    def forward(
            self,
            input_features: torch.Tensor,
            actions: Union[torch.Tensor, None] = None):
        """
        Forward pass in all the networks (actor and critic)

        :param input_features:
        :param actions:
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf = self.extractor(input_features)
        distribution = self.output_action_distribution(latent_pi)
        if latent_vf is None:
            latent_vf = latent_pi
        value = self.value_net(latent_vf)
        if actions is None:
            return distribution, value, None
        else:
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()
            return value, log_prob, entropy


class multi_agent_ACP():

    def __init__(
            self,
            vehicle_num: int,
            weight_shape: int,
            share_policy: bool,
            ortho_init: bool,
            conv_params: list,
            add_BN: bool,
            output_dim: list,
            share_params: bool,
            action_dim: int,
            learning_rate: Union[int, float]):
        '''
        :param conv_params: with input layer params
        :param output_dim: without input dim
        :return:
        '''

        self.vehicle_num = vehicle_num
        self.share_policy = share_policy

        conv_output_shape = weight_shape
        for params in conv_params:
            if 'padding' not in params:
                if type(conv_output_shape) == tuple:
                    params['padding'] = (0, 0)
                else:
                    params['padding'] = 0
            if 'dilation' not in params:
                if type(conv_output_shape) == tuple:
                    params['dilation'] = (1, 1)
                else:
                    params['dilation'] = 1
            if 'stride' not in params:
                if type(conv_output_shape) == tuple:
                    params['stride'] = (1, 1)
                else:
                    params['stride'] = 1
            conv_output_shape = self.cal_conv_output_shape(
                input_shape=conv_output_shape,
                kernel_size=params['kernel_size'],
                padding=params['padding'],
                dilation=params['dilation'],
                stride=params['stride'],
            )
        params = conv_params[-1]
        if type(conv_output_shape) == tuple:
            output_dim = [conv_output_shape[0] * conv_output_shape[1] *
                          params['out_channels']] + output_dim
        else:
            output_dim = [conv_output_shape * params['out_channels']] + output_dim

        self.ACP = {}
        for i in range(vehicle_num):
            if i == 0:
                self.ACP[i] = ActorCriticPolicy(
                    ortho_init=ortho_init,
                    conv_params=conv_params,
                    add_BN=add_BN,
                    output_dim=output_dim,
                    share_params=share_params,
                    action_dim=action_dim,
                    learning_rate=learning_rate,
                )
            else:
                if share_policy:
                    self.ACP[i] = self.ACP[0]
                else:
                    self.ACP[i] = ActorCriticPolicy(
                        ortho_init=ortho_init,
                        conv_params=conv_params,
                        add_BN=add_BN,
                        output_dim=output_dim,
                        share_params=share_params,
                        action_dim=action_dim,
                        learning_rate=learning_rate,
                    )

        self.device = torch.device('cpu')

    @staticmethod
    def cal_conv_output_shape(input_shape, kernel_size, padding, dilation, stride):
        if type(input_shape) == tuple:
            row = int((input_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
            col = int((input_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
            return (row, col)
        else:
            return int((input_shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def to(self, param):
        for i in range(self.vehicle_num):
            self.ACP[i].to(param)
        if param == 'cuda':
            self.device = torch.device('cuda')
        elif param == torch.device('cuda'):
            self.device = torch.device('cuda')
        elif param == 'cpu':
            self.device = torch.device('cpu')
        elif param == torch.device('cpu'):
            self.device = torch.device('cpu')
        return self

    def train(self):
        for i in range(self.vehicle_num):
            self.ACP[i].train()
        return self

    def eval(self):
        for i in range(self.vehicle_num):
            self.ACP[i].eval()
        return self

    def state_dict(self):
        ACP_params = {}
        for key in self.ACP.keys():
            if (self.share_policy and key == 0) or (not self.share_policy):
                ACP_params[key] = self.ACP[key].state_dict()
        return ACP_params

    def load_state_dict(self, ACP_params):
        for key in self.ACP.keys():
            if (self.share_policy and key == 0) or (not self.share_policy):
                self.ACP[key].load_state_dict(ACP_params[key])

    def optimize(self, loss: torch.Tensor, max_grad_norm):

        if self.share_policy:
            self.ACP[0].optimizer.zero_grad()
        else:
            for i in range(self.vehicle_num):
                self.ACP[i].optimizer.zero_grad()
        loss.backward()
        if self.share_policy:
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.ACP[0].parameters(), max_grad_norm)
            self.ACP[0].optimizer.step()
        else:
            for i in range(self.vehicle_num):
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.ACP[i].parameters(), max_grad_norm)
                self.ACP[i].optimizer.step()

    def forward(
            self,
            loc_features: np.ndarray,
            weight_features: np.ndarray,
            actions: Union[np.ndarray, None] = None):
        """
        Forward pass in all the networks (actor and critic)

        :param loc_features:
        :param weight_features:
        :return: action, value and log probability of the action
        """
        loc_features = loc_features.reshape((loc_features.shape[0], -1, 1))
        index1 = np.repeat(np.array(range(loc_features.shape[0])), self.vehicle_num * 2, axis=0)
        index2 = np.array([0] * len(index1))
        index3 = loc_features[:, :, 0].flatten()
        index = np.vstack((index1, index2, index3))
        input_all_loc_features = torch.zeros(weight_features.shape, dtype=torch.float32, device=self.device)
        input_all_loc_features[index] = 1

        weight_features = torch.as_tensor(weight_features, dtype=torch.float32, device=self.device)
        if actions is not None:
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device).long()

        values = []
        distributions = []
        log_probs = []
        entropys = []

        for i in range(self.vehicle_num):

            vehicle_index = index[:, [i * 2, i * 2 + 1]]
            input_loc_features = torch.zeros(weight_features.shape, dtype=torch.float32, device=self.device)
            input_loc_features[vehicle_index] = 1
            input_features = torch.cat((input_loc_features, input_all_loc_features, weight_features), dim=1)

            if actions is None:
                distribution, value, _ = self.ACP[i](input_features, None)
                distributions.append(copy.deepcopy(distribution))
            else:
                value, log_prob, entropy = self.ACP[i](input_features, actions[:, i])
                log_probs.append(log_prob.view(log_prob.shape + (1,)).clone())
                entropys.append(entropy.view(log_prob.shape + (1,)).clone())
            values.append(value.clone())

        values = torch.cat(values, dim=1)
        if actions is None:
            return distributions, values, None
        else:
            log_probs = torch.cat(log_probs, dim=1)
            entropys = torch.cat(entropys, dim=1)
            return values, log_probs, entropys


class QLearningPolicy(nn.Module):

    def __init__(
        self,
        conv_params: list,
        add_BN: bool,
        output_dim: list,
        action_dim: int,
        learning_rate: Union[int, float],
    ):
        '''
        :param conv_params: type of item in iteration must be dict object, and dict keys are Conv layer
        param names, key values are allowed param values
        :param add_BN:
        :param output_dim: type of item in iteration must be int object
        :param action_dim:
        :param learning_rate:
        :return:
        '''
        super(QLearningPolicy, self).__init__()

        self.extractor = Conv1dExtractor(
            conv_params=conv_params,
            add_BN=add_BN,
            output_dim=output_dim,
            share_params=True,
        )
        self.value_net = nn.Linear(output_dim[-1], action_dim)

        # Setup optimizer with initial learning rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

    def forward(self, input_features: torch.Tensor):
        """
        Forward pass in network

        :param input_features:
        :return:
        """
        latent_vf, _ = self.extractor(input_features)
        value = self.value_net(latent_vf)
        return value


class multi_agent_QLP():

    def __init__(
            self,
            vehicle_num: int,
            weight_shape: int,
            share_policy: bool,
            conv_params: list,
            add_BN: bool,
            output_dim: list,
            action_dim: int,
            learning_rate: Union[int, float]):
        '''
        :param conv_params: with input layer params
        :param output_dim: without input dim
        :return:
        '''

        self.vehicle_num = vehicle_num
        self.share_policy = share_policy

        conv_output_shape = weight_shape
        for params in conv_params:
            if 'padding' not in params:
                if type(conv_output_shape) == tuple:
                    params['padding'] = (0, 0)
                else:
                    params['padding'] = 0
            if 'dilation' not in params:
                if type(conv_output_shape) == tuple:
                    params['dilation'] = (1, 1)
                else:
                    params['dilation'] = 1
            if 'stride' not in params:
                if type(conv_output_shape) == tuple:
                    params['stride'] = (1, 1)
                else:
                    params['stride'] = 1
            conv_output_shape = self.cal_conv_output_shape(
                input_shape=conv_output_shape,
                kernel_size=params['kernel_size'],
                padding=params['padding'],
                dilation=params['dilation'],
                stride=params['stride'],
            )
        params = conv_params[-1]
        if type(conv_output_shape) == tuple:
            output_dim = [conv_output_shape[0] * conv_output_shape[1] *
                          params['out_channels']] + output_dim
        else:
            output_dim = [conv_output_shape * params['out_channels']] + output_dim

        self.QLP = {}
        for i in range(vehicle_num):
            if i == 0:
                self.QLP[i] = QLearningPolicy(
                    conv_params=conv_params,
                    add_BN=add_BN,
                    output_dim=output_dim,
                    action_dim=action_dim,
                    learning_rate=learning_rate,
                )
            else:
                if share_policy:
                    self.QLP[i] = self.QLP[0]
                else:
                    self.QLP[i] = QLearningPolicy(
                        conv_params=conv_params,
                        add_BN=add_BN,
                        output_dim=output_dim,
                        action_dim=action_dim,
                        learning_rate=learning_rate,
                    )

        self.device = torch.device('cpu')

    @staticmethod
    def cal_conv_output_shape(input_shape, kernel_size, padding, dilation, stride):
        if type(input_shape) == tuple:
            row = int((input_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
            col = int((input_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
            return (row, col)
        else:
            return int((input_shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def to(self, param):
        for i in range(self.vehicle_num):
            self.QLP[i].to(param)
        if param == 'cuda':
            self.device = torch.device('cuda')
        elif param == torch.device('cuda'):
            self.device = torch.device('cuda')
        elif param == 'cpu':
            self.device = torch.device('cpu')
        elif param == torch.device('cpu'):
            self.device = torch.device('cpu')
        return self

    def train(self):
        for i in range(self.vehicle_num):
            self.QLP[i].train()
        return self

    def eval(self):
        for i in range(self.vehicle_num):
            self.QLP[i].eval()
        return self

    def state_dict(self):
        QLP_params = {}
        for key in self.QLP.keys():
            if (self.share_policy and key == 0) or (not self.share_policy):
                QLP_params[key] = self.QLP[key].state_dict()
        return QLP_params

    def load_state_dict(self, QLP_params):
        for key in self.QLP.keys():
            if (self.share_policy and key == 0) or (not self.share_policy):
                self.QLP[key].load_state_dict(QLP_params[key])

    def optimize(self, loss: torch.Tensor, max_grad_norm):

        if self.share_policy:
            self.QLP[0].optimizer.zero_grad()
        else:
            for i in range(self.vehicle_num):
                self.QLP[i].optimizer.zero_grad()
        loss.backward()
        if self.share_policy:
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.QLP[0].parameters(), max_grad_norm)
            self.QLP[0].optimizer.step()
        else:
            for i in range(self.vehicle_num):
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.QLP[i].parameters(), max_grad_norm)
                self.QLP[i].optimizer.step()

    def forward(self, loc_features: np.ndarray, weight_features: np.ndarray):
        """
        Forward pass in network

        :param loc_features:
        :param weight_features:
        :return:
        """
        loc_features = loc_features.reshape((loc_features.shape[0], -1, 1))
        index1 = np.repeat(np.array(range(loc_features.shape[0])), self.vehicle_num * 2, axis=0)
        index2 = np.array([0] * len(index1))
        index3 = loc_features[:, :, 0].flatten()
        index = np.vstack((index1, index2, index3))
        input_all_loc_features = torch.zeros(weight_features.shape, dtype=torch.float32, device=self.device)
        input_all_loc_features[index] = 1

        weight_features = torch.as_tensor(weight_features, dtype=torch.float32, device=self.device)

        values = []
        for i in range(self.vehicle_num):
            vehicle_index = index[:, [i * 2, i * 2 + 1]]
            input_loc_features = torch.zeros(weight_features.shape, dtype=torch.float32, device=self.device)
            input_loc_features[vehicle_index] = 1
            input_features = torch.cat((input_loc_features, input_all_loc_features, weight_features), dim=1)
            value = self.QLP[i](input_features).unsqueeze(1)
            values.append(value.clone())
        values = torch.cat(values, dim=1)
        return values


if __name__ == '__main__':

    vehicle_num = 50
    weight_shape = 20 * 20
    conv_params = [{
        'in_channels': 3,
        'out_channels': 1,
        'kernel_size': 3,
        'stride': 2,
        'padding': 1,
        'dilation': 1,
    }]

    maacp = multi_agent_ACP(
        vehicle_num=vehicle_num,
        weight_shape=weight_shape,
        share_policy=True,
        ortho_init=True,
        conv_params=conv_params,
        add_BN=True,
        output_dim=[32],
        share_params=False,
        action_dim=4,
        learning_rate=0.00001
    )

    maqlp = multi_agent_QLP(
        vehicle_num=vehicle_num,
        weight_shape=weight_shape,
        share_policy=True,
        conv_params=conv_params,
        add_BN=True,
        output_dim=[32],
        action_dim=4,
        learning_rate=0.00001
    )

    # test decision making
    bacth_size = 1
    loc_features = np.random.rand(bacth_size, vehicle_num * 2)
    weight_features = np.random.rand(bacth_size, 1, weight_shape)
    values_ = maqlp.forward(
        loc_features=loc_features,
        weight_features=weight_features,
    )
    print(values_)
    actions = None
    distributions, values, _ = maacp.forward(
        loc_features=loc_features,
        weight_features=weight_features,
        actions=actions,
    )
    print(distributions)
    print(distributions[0].get_actions())
    print(distributions[1].get_actions())
    print(distributions[1].all_probs())
    print(values)

    # test action probs
    bacth_size = 100
    loc_features = np.random.rand(bacth_size, vehicle_num * 2)
    weight_features = np.random.rand(bacth_size, 1, weight_shape)
    actions = np.random.randint(low=0, high=4, size=(bacth_size, vehicle_num))
    print(actions)
    values, log_probs, entropys = maacp.forward(
        loc_features=loc_features,
        weight_features=weight_features,
        actions=actions,
    )
    print(values)
    print(log_probs)
    print(entropys)