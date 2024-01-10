import sys
import os
import copy
from functools import partial
from typing import Tuple

import numpy as np
from torch.distributions import Categorical

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_path)

from grid_score_base_project.multi_agent_dispatching.MAGPPO.model import *


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
        in_channels: int,
        action_dim: int,
        learning_rate: Union[int, float],
    ):
        '''
        :param ortho_init:
        :param action_dim:
        :param learning_rate:
        :return:
        '''
        super(ActorCriticPolicy, self).__init__()

        self.ortho_init = ortho_init

        self.extractor = GCN_S2S_Extractor(in_channels=in_channels)
        self.value_net = nn.Linear(32, 1)
        # Action distribution
        self.action_distribution = CategoricalDistribution(action_dim=action_dim)
        self.action_net = self.action_distribution.proba_distribution_net(latent_dim=32)

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
        if isinstance(module, (nn.Linear, GCNConv)):
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
            x: torch.Tensor,
            edge_index: Union[torch.Tensor, torch_sparse.tensor.SparseTensor],
            edge_weight: Optional[torch.Tensor],
            batch: torch.LongTensor,
            actions: Union[torch.Tensor, None] = None):
        """
        Forward pass in all the networks (actor and critic)

        :param x:
        :param edge_index:
        :param edge_weight:
        :param batch:
        :param actions:
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf = self.extractor(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
        )
        distribution = self.output_action_distribution(latent_pi)
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
            ortho_init: bool,
            in_channels: int,
            action_dim: int,
            learning_rate: Union[int, float]):
        '''
        :param vehicle_num:
        :param ortho_init:
        :param in_channels:
        :param action_dim:
        :param learning_rate:
        :return:
        '''

        self.vehicle_num = vehicle_num

        self.ACP = ActorCriticPolicy(
            ortho_init=ortho_init,
            in_channels=in_channels,
            action_dim=action_dim,
            learning_rate=learning_rate
        )

        self.device = torch.device('cpu')

    def to(self, param):
        self.ACP.to(param)
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
        self.ACP.train()
        return self

    def eval(self):
        self.ACP.eval()
        return self

    def state_dict(self):
        ACP_params = copy.deepcopy(self.ACP.state_dict())
        for key in ACP_params.keys():
            ACP_params[key] = ACP_params[key].to('cpu')
        return ACP_params

    def load_state_dict(self, ACP_params):
        self.ACP.load_state_dict(ACP_params)

    def optimize(self, loss: torch.Tensor, max_grad_norm):
        self.ACP.optimizer.zero_grad()
        loss.backward()
        # Clip grad norm
        torch.nn.utils.clip_grad_norm_(self.ACP.parameters(), max_grad_norm)
        self.ACP.optimizer.step()

    def forward(
            self,
            edge_index: Union[torch.Tensor, torch_sparse.tensor.SparseTensor],
            edge_weight: Optional[torch.Tensor],
            batch: torch.Tensor,
            edge_loc_features: Union[torch.Tensor, None] = None,
            the_same_features: Union[torch.Tensor, None] = None,
            state_features: Union[torch.Tensor, None] = None,
            actions: Union[torch.Tensor, None] = None,
    ):
        """
        Forward pass in all the networks (actor and critic)

        :param edge_index:
        :param edge_weight:
        :param batch:
        :param edge_loc_features:
        :param the_same_features:
        :param state_features:
        :param actions:
        :return: action, value and log probability of the action
        """
        values = []
        distributions = []
        if actions is None:
            # It seems to be able to calculate one time by concatenating all vehicle state features
            for i in range(self.vehicle_num):
                edge_loc_feature = edge_loc_features[i]
                input_features = torch.cat((edge_loc_feature, the_same_features), dim=1).to(self.device)
                distribution, value, _ = self.ACP(
                    input_features, edge_index.to(self.device), edge_weight.to(self.device),
                    batch.to(self.device), None
                )
                distributions.append(copy.deepcopy(distribution))
                values.append(value.clone())
            values = torch.cat(values, dim=1)
            return distributions, values, None
        else:
            values, log_probs, entropy = self.ACP(
                state_features.to(self.device), edge_index.to(self.device), edge_weight.to(self.device),
                batch.to(self.device), actions.to(self.device)
            )
            log_probs = log_probs.view(log_probs.shape + (1,))
            entropy = entropy.view(entropy.shape + (1,))
            return values, log_probs, entropy


if __name__ == '__main__':

    vehicle_num = 50
    in_channels = 8

    maacp = multi_agent_ACP(
        vehicle_num=vehicle_num,
        ortho_init=True,
        in_channels=in_channels,
        action_dim=8,
        learning_rate=0.00001
    )
    maacp.eval()

    # test decision making
    ad_mat = torch.zeros((10, 10), dtype=torch.int32)
    ad_mat[(torch.randint(0, 10, (20, )), torch.randint(0, 10, (20, )))] = 1
    edge_index = torch.stack(torch.where(ad_mat), dim=1).T
    edge_weight = torch.randint(1, 5, (edge_index.shape[1], )).to(torch.float32)
    batch = torch.zeros(10, dtype=torch.long)
    # edge_loc_features = torch.from_numpy(np.random.rand(vehicle_num, 10, 1).astype(np.float32))
    # the_same_features = torch.from_numpy(np.random.rand(10, 7).astype(np.float32))
    edge_loc_features = torch.zeros((vehicle_num, 10, 1), dtype=torch.float32)
    edge_loc_features[1, 1, 0] = 1
    the_same_features = torch.zeros((10, 7), dtype=torch.float32)
    distributions, values, _ = maacp.forward(
        edge_index=edge_index,
        edge_weight=edge_weight,
        batch=batch,
        edge_loc_features=edge_loc_features,
        the_same_features=the_same_features,
        actions=None,
    )
    # print(distributions)
    # print(distributions[0].get_actions())
    # print(distributions[1].get_actions())
    print(distributions[1].all_probs())
    # print(values.shape)

    # # test action probs
    # ad_mat = torch.zeros((10, 10), dtype=torch.int32)
    # ad_mat[(torch.randint(0, 10, (20,)), torch.randint(0, 10, (20,)))] = 1
    # edge_index = torch.stack(torch.where(ad_mat == 1), dim=1).T
    # edge_weight = torch.randint(1, 5, (edge_index.shape[1],)).to(torch.float32)
    # batch = torch.zeros(10, dtype=torch.long)
    # edge_loc_features = np.random.rand(10, 1).astype(np.float32)
    # the_same_features = np.random.rand(10, 7).astype(np.float32)
    # state_features = torch.from_numpy(np.concatenate((edge_loc_features, the_same_features), axis=1))
    # actions = np.random.randint(low=0, high=4, size=(1,))
    # actions = torch.as_tensor(actions, dtype=torch.long)
    # print(actions)
    # values, log_probs, entropy = maacp.forward(
    #     edge_index=edge_index,
    #     edge_weight=edge_weight,
    #     batch=batch,
    #     state_features=state_features,
    #     actions=actions,
    # )
    # print(values)
    # print(log_probs)
    # print(entropy)