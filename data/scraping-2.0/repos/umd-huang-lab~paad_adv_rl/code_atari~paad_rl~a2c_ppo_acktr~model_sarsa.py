import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from paad_rl.a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, Beta
from paad_rl.a2c_ppo_acktr.utils import init

STD = 2**0.5

def initialize_weights(mod, initialization_type, scale=STD):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    '''
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")

def orthogonal_init(tensor, gain=1):
    '''
    Fills the input `Tensor` using the orthogonal initialization scheme from OpenAI
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> orthogonal_init(w)
    '''
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor

def mlp(sizes, activation, output_activation=nn.Identity, init="orthogonal"):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        l = nn.Linear(sizes[j], sizes[j+1])
        if j == len(sizes)-1:
            initialize_weights(l, init, 1.0)
        else:
            initialize_weights(l, init)
        layers += [l, act()]
    return nn.Sequential(*layers)

class CNN_Layers(nn.Module):
    hidden_size=512
    def __init__(self, in_channels=4, num_actions=18):
        super(CNN_Layers, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return x

class QFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, atari=False):
        super().__init__()
        self.cnn_layers = CNN_layers(num_actions=act_dim) if atari else None
        if atari:
            self.q = mlp([CNN_layers.hidden_size + act_dim] + list(hidden_sizes) + [1], activation)
        else:
            self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        if (self.cnn_layers is not None):
            obs = self.cnn_layers(obs)
        q = self.q(torch.cat([obs, act], dim=-1))
        return q
        # return torch.squeeze(q, -1) # Critical to ensure q has right shape.


