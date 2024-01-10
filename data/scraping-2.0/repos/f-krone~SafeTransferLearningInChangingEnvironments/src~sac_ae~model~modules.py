import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from .misc import *


OUT_DIM = {2: 39, 4: 35, 6: 31}
LOG_FREQ = 1000


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)


class SharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32, stride=1, cnn_3dconv=False):
        super().__init__()
        if cnn_3dconv:
            assert len(obs_shape) == 4, 'Expected obs shape to have length 4, but got ' + str(obs_shape)
        else:
            assert len(obs_shape) == 3, 'Expected obs shape to have length 3, but got ' + str(obs_shape)
        self.num_layers = num_layers
        self.num_filters = num_filters

        # self.layers = nn.Sequential(
        #     nn.Conv2d(obs_shape[0], 16, 3, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 3, stride=2),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )

        if cnn_3dconv:
            self.layers = nn.Sequential(
                    nn.Conv3d(obs_shape[0], 16, kernel_size=3, stride = 2, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(16, 32, kernel_size=3, stride = 2, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(32, 64, kernel_size=3, stride = (1, 2, 2), padding=1),
                    nn.ReLU(),
                    nn.Flatten()
                )
        else:
            self.layers = [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
            self.layers.append(nn.ReLU())
            for _ in range(1, num_layers):
                self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=stride))
                self.layers.append(nn.ReLU())
            self.layers.append(Flatten())
            self.layers = nn.Sequential(*self.layers)

        self.out_dim = get_out_shape(obs_shape, self.layers)[-1]
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x/255.)

class ATCSharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=4, num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_layers = num_layers
        self.num_filters = num_filters

        # strides of [2,2,2,1] with final nonlinear layer
        self.layers = [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        self.layers.append(nn.ReLU())
        for _ in range(1, num_layers-1):
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=2))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers.append(nn.ReLU())
        self.layers.append(Flatten())
        
        self.layers = nn.Sequential(*self.layers)

        self.out_dim = get_out_shape(obs_shape, self.layers)[-1]
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x/255.)

class RLProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
        self.out_dim = out_dim
        self.apply(weight_init)
    
    def forward(self, x):
        return self.projection(x)


class PlainProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )
        self.out_dim = out_dim
        self.apply(weight_init)
    
    def forward(self, x):
        return self.projection(x)    

class RobotEncoder(nn.Module):
    def __init__(self, in_dim, architecture):
        super().__init__()
        self.out_dim = architecture[-1]
        self.mlp = []
        out_dim = in_dim
        for dim in architecture:
            self.mlp.append(nn.Linear(out_dim, dim))
            self.mlp.append(nn.ReLU())
            out_dim = dim
        self.mlp.insert(len(self.mlp)-1, nn.LayerNorm(out_dim))
        self.mlp = nn.Sequential(*self.mlp)
        self.apply(weight_init)
    
    def forward(self, x):
        return self.mlp(x)

class Encoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, cnn, projection):
        super().__init__()
        self.cnn = cnn
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x, detach=False):
        x = self.cnn(x)
        if detach:
            x = x.detach()
        return self.projection(x)

class StateEncoder(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.projection = projection

    def forward(self, x, detach=False):
        if detach:
            x = x.detach()
        return self.projection(x)

class Decoder(nn.Module):
    def __init__(self, num_channels, feature_dim, num_layers = 4, num_filters = 32, cnn_3dconv=False, stack_size=3, image_size=84):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]
        self.cnn_3dconv = cnn_3dconv
        self.stack_size = stack_size
        self.image_size = image_size

        # if self.cnn_3dconv:
        #     self.fc = nn.Linear(feature_dim, 64*1*11*11)

        #     self.deconvs = nn.ModuleList()
        #     self.deconvs.append(nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, output_padding=0))
        #     self.deconvs.append(nn.ConvTranspose3d(32, 16, kernel_size=3, stride=(2, 2, 2), padding=(2, 2, 2), output_padding=0))
        #     self.deconvs.append(nn.ConvTranspose3d(16, num_channels, kernel_size=3, stride=(2, 2, 2), padding=(2, 2, 2), output_padding=(0, 1, 1)))
        # else:
        self.fc = nn.Linear(feature_dim, num_filters * self.out_dim * self.out_dim)

        self.deconvs = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.deconvs.append(nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1))
        self.deconvs.append(nn.ConvTranspose2d(num_filters, num_channels, 3, stride=2, output_padding=1))


    def forward(self, h):
        h = torch.relu(self.fc(h))
        # if self.cnn_3dconv:
        #     x = h.view(-1, 64, 1, 11, 11)
        # else:
        x = h.view(-1, self.num_filters, self.out_dim, self.out_dim)

        for i in range(0,len(self.deconvs) - 1):
            x = torch.relu(self.deconvs[i](x))
        obs = self.deconvs[-1](x)
        if self.cnn_3dconv:
            obs = obs.view(-1, 3, self.stack_size, self.image_size, self.image_size)
        return obs


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(self, encoder, action_dim, hidden_dim, log_std_min, log_std_max, robot_shape, robot_encoder):
        super().__init__()
        self.encoder = encoder
        self.robot_encoder = robot_encoder
        self.robot = robot_shape > 0
        self.mlp = []
        out_dim = self.encoder.out_dim
        if self.robot_encoder != None:
            out_dim += self.robot_encoder.out_dim
        else:
            out_dim += robot_shape
        for dim in hidden_dim:
            self.mlp.append(nn.Linear(out_dim, dim))
            self.mlp.append(nn.LayerNorm(dim))
            self.mlp.append(nn.ReLU())
            out_dim = dim
        self.mlp.append(nn.Linear(out_dim, 2 * action_dim))
        self.mlp = nn.Sequential(*self.mlp)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.apply(weight_init)

    def forward(self, x, compute_pi=True, compute_log_pi=True, detach=False):
        if self.robot:
            image = x['image']
            robot_obs = x['robot']

            if self.robot_encoder != None:
                robot_obs = self.robot_encoder(robot_obs)

            image = self.encoder(image, detach=detach)
            x = torch.cat([image, robot_obs], dim=1)
        else:
            x = self.encoder(x, detach=detach)
        mu, log_std = self.mlp(x).chunk(2, dim=-1)
        # taken from openai/spinningup
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        # pi_distribution = Normal(mu, std)
        pi_distribution = SquashedNormal(mu, std)
        if compute_pi or compute_log_pi:
            pi = pi_distribution.rsample()
        else:
            pi = None

        if compute_log_pi:
            log_pi = pi_distribution.log_prob(pi).sum(axis=-1, keepdim=True)
            # log_pi -= (2*(np.log(2) - pi - F.softplus(-2*pi))).sum(axis=1, keepdim=True)
        else:
            log_pi = None
        mu = torch.tanh(mu)
        return mu, pi, log_pi, log_std
    

class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.mlp = []
        out_dim = obs_dim + action_dim
        for dim in hidden_dim:
            self.mlp.append(nn.Linear(out_dim, dim))
            self.mlp.append(nn.LayerNorm(dim))
            self.mlp.append(nn.ReLU())
            out_dim = dim
        self.mlp.append(nn.Linear(out_dim, 1))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=1)
        return self.mlp(obs_action)


class Critic(nn.Module):
    def __init__(self, encoder, action_dim, hidden_dim, robot_shape, robot_encoder):
        super().__init__()
        self.robot = robot_shape > 0
        self.robot_encoder = robot_encoder
        self.encoder = encoder
        in_dim = self.encoder.out_dim
        if self.robot_encoder != None:
            in_dim += self.robot_encoder.out_dim
        else:
            in_dim += robot_shape
        self.Q1 = QFunction(in_dim, action_dim, hidden_dim)
        self.Q2 = QFunction(in_dim, action_dim, hidden_dim)
        self.apply(weight_init)

    def forward(self, x, action, detach=False):
        if self.robot:
            image = self.encoder(x['image'], detach=detach)
            robot = x['robot']
            if self.robot_encoder != None:
                robot = self.robot_encoder(robot)
            x = torch.cat([image, robot], dim=1)
            return self.Q1(x, action), self.Q2(x, action)
        else:
            x = self.encoder(x, detach=detach)
            return self.Q1(x, action), self.Q2(x, action)


class CURL(nn.Module):
    def __init__(self, encoder):
        super(CURL, self).__init__()
        self.encoder = encoder
        self.W = nn.Parameter(torch.rand(encoder.out_dim, encoder.out_dim))

    def encode(self, x):
        return self.encoder(x)

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, robot):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.robot = robot

    def recon(self, x):
        h = self.encoder(x)
        recon_x = self.decoder(h)
        return recon_x
    
    
class ATC(nn.Module):
    def __init__(self, encoder, atc_hidden_feature_dim):
        super(ATC, self).__init__()
        self.encoder = encoder
        self.W = nn.Parameter(torch.rand(encoder.out_dim, encoder.out_dim))

        self.anchor_mlp = nn.Sequential(
            nn.Linear(encoder.out_dim, atc_hidden_feature_dim), nn.ReLU(),
            nn.Linear(atc_hidden_feature_dim, encoder.out_dim))

    def encode(self, x):
        z = self.encoder(x)
        return z + self.anchor_mlp(z)

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits