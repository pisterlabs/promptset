# This code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Licence.  The full licence and information is available at:
# https://github.com/plai-group/a2d/blob/master/LICENCE.md
# © Andrew Warrington, J. Wilder Lavington, Adam Scibior, Mark Schmidt & Frank Wood.
# Accompanies the paper: Robust Asymmetric Learning in POMDPs, ICML 2021.

import torch.nn as nn
import math
import torch as ch
import torch.nn.functional as F
from a2d.util.torch_utils import *
from .encoder import PixelEncoder, Identity


def initialize_weights_scalable(mod, initialization_type, scale=2**0.5):
    """
    Slightly more involved initialization that allows us to scale the layers.
    This can improve the stability of policy learning in certain scenarios.
    :param mod:
    :param initialization_type:
    :param scale:
    :return:
    """
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


def initialize_weights(m):
    """
    Custom weight init for Conv2D and Linear layers.
    :param m:
    :return:
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim, output_dim, hidden_size=64, hidden_depth=2, activ=nn.Tanh, output_mod=None):
    """

    :param input_dim:
    :param output_dim:
    :param hidden_size:
    :param hidden_depth:
    :param activ:
    :param output_mod:
    :return:
    """
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_size), activ()]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_size, hidden_size), activ()]
        mods.append(nn.Linear(hidden_size, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


# def apply(func, M):
#     if len(M.size()) == 1:
#         M=M.unsqueeze(0)
#     elif len(M.size()) == 2:
#         pass
#     elif len(M.size()) == 3:
#         M=M.reshape((M.size()[0]*M.size()[1], M.size()[2]))
#     else:
#         raise Exception('incorrect shaping')
#     tList = [func(m) for m in ch.unbind(M, dim=0) ]
#     res = ch.stack(tList, dim=0)
#     return res


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
    u, s, v = ch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with ch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


class ValueDenseNet(nn.Module):
    """
    Value network class.
    """

    def __init__(self, state_dim, init='orthogonal', activ=nn.Tanh, hidden_depth=2, hidden_size=64, slim_init=True):
        """
        Initialize the value network.
        :param state_dim:
        :param init:
        :param activ:
        :param hidden_depth:
        :param hidden_size:
        :param slim_init:
        """
        super().__init__()
        self.DTYPE = ch.double
        self.obs_shape = state_dim

        # Pretty ugly bit of code for parsing input sizes.
        try:
            # Catch if it is a length one tuple.
            isvector = len(state_dim) == 1
            if isvector:
                state_dim = state_dim[0]
            else:
                assert False  # trigger try-catch manually.
        except:
            try:
                # Catch if it is just an int
                isvector = isinstance(state_dim, int)
            except:
                isvector = False

        if isvector:  # isinstance(state_dim, int):
            self.type = 'flat'
            self.encoder = Identity(state_dim)
            self.head = mlp(state_dim, 1, hidden_size=hidden_size,
                            hidden_depth=hidden_depth, activ=activ, output_mod=None)

            if init == 'fixed':
                initialize_weights(self.head[:-1])
            else:
                initialize_weights_scalable(self.head[:-1], init)
                if slim_init:
                    initialize_weights_scalable(self.head[-1], init, scale=0.01)
                else:
                    initialize_weights_scalable(self.head[-1], init)

            self.trunk = nn.Sequential(self.encoder, self.head)

        elif len(state_dim) == 3:  # isinstance(state_dim, tuple):
            self.type = 'image'
            encoder_input = state_dim
            self.encoder = PixelEncoder(ch.tensor(encoder_input))
            self.head = mlp(self.encoder.feature_dim, 1, hidden_size=hidden_size,
                            hidden_depth=hidden_depth, activ=activ, output_mod=None)

            if init == 'fixed':
                initialize_weights(self.head)
                initialize_weights(self.head[-1])
            else:
                initialize_weights_scalable(self.head, init)
                initialize_weights_scalable(self.head[-1], init, scale=1.0)

            self.trunk = nn.Sequential(self.encoder, self.head)

        else:
            raise NotImplementedError

        self.to(self.DTYPE)

        # Little switch flag to make sure that we use the correct state/obs sometimes.
        self.encoder.PRETRAINED = False
        self.EXTERNAL = False
        self.ASYMMETRIC = False


    def _do_flat(self, x):
        """
        Flatten/reshape the inputs to the network as appropriate.
        :param x:   (tensor):   states to evaluate at.
        :return:    (tensor):   evaluations of network.
        """
        # If we are using flat vectors, then make sure we flatten out the dimensions.
        if self.type == 'flat':
            x = x.reshape((x.shape[0], -1))
        else:
            # If we are using images, we need to sort the dimensions out a bit.
            x = x.permute(1, 4, 0, 2, 3)
            x = ch.cat([_o for _o in x], dim=0)
            x = x.permute(1, 0, 2, 3)
        return x


    def forward(self, x):
        """
        Evaluate the network at states x
        :param x:   (tensor):   states to evaluate at.
        :return:    (tensor):   value function evaluations at states x
        """
        x = x.type(next(self.parameters()).type())
        x = self._do_flat(x)
        value = self.trunk(x)
        return value.squeeze()


    def get_value(self, x, detach_encoder=None):
        """
        Wraps call to ```value'''
        :param x:               (tensor):   states to evaluate at.
        :param detach_encoder:  (bool):     no longer used.
        :return:                (tensor):   value function evaluations at states x.
        """
        return self.value(x, detach_encoder=detach_encoder)


    def value(self, x, detach_encoder=None):
        """
        Call the value function at states x.
        :param x:               (tensor):   states to evaluate at.
        :param detach_encoder:  (bool):     no longer used.
        :return:                (tensor):   value function evaluations at states x.
        """
        return self(x)


class DiscPolicy(nn.Module):
    """
    Define a discrete policy network.
    """

    def __init__(self, state_dim, action_dim, activ=nn.Tanh, hidden_depth=2, hidden_size=64,
                 init='orthogonal', _encoder_dim=None):
        """
        Initialize the policy.

        :param state_dim:
        :param action_dim:
        :param activ:
        :param hidden_depth:
        :param hidden_size:
        :param init:
        :param _encoder_dim:
        """
        super().__init__()
        self.DTYPE = ch.double
        self.discrete = True
        self.obs_shape = state_dim
        self.num_outputs = action_dim
        self.num_actions = action_dim

        if _encoder_dim is None:
            _encoder_dim = hidden_size

        # Pretty ugly bit of code for parsing input sizes and working out shapes.
        try:
            # Catch if it is a length one tuple.
            isvector = len(state_dim) == 1
            if isvector:
                state_dim = state_dim[0]
            else:
                assert False  # trigger try-catch manually.
        except:
            try:
                # Catch if it is just an int
                isvector = isinstance(state_dim, int)
            except:
                isvector = False


        # If it is a vector, then it is a flat input, as opposed to an image.
        if isvector:
            self.type = 'flat'
            self.encoder = Identity(state_dim)
            self.head = mlp(state_dim, action_dim, hidden_size=hidden_size,
                            hidden_depth=hidden_depth, activ=activ, output_mod=None)

            if init == 'fixed':
                initialize_weights(self.head[:-1])
                initialize_weights(self.head[-1])
            else:
                initialize_weights_scalable(self.head[:-1], init)
                initialize_weights_scalable(self.head[-1], init, scale=0.01)
            self.trunk = nn.Sequential(self.encoder, self.head)

        elif len(state_dim) == 3:
            self.type = 'image'
            self.encoder = PixelEncoder(ch.tensor(state_dim), feature_dim=_encoder_dim)
            self.head = mlp(self.encoder.feature_dim, action_dim, hidden_size=hidden_size,
                            hidden_depth=hidden_depth, activ=activ, output_mod=None)

            if init == 'fixed':
                initialize_weights(self.head)
                initialize_weights(self.head[-1])
            else:
                initialize_weights_scalable(self.head, init)
                initialize_weights_scalable(self.head[-1], init, scale=1.0)

            self.trunk = nn.Sequential(self.encoder, self.head)

        else:
            raise NotImplementedError

        self.to(self.DTYPE)

        # Little switch flags to make sure that we use the correct state/obs sometimes.
        self.encoder.PRETRAINED = False     # If the encoder is pretrained we won't apply gradient updates to it.
        self.EXTERNAL = False               # This is no longer used.
        self.ASYMMETRIC = False             # This is no longer used. (It is used in ValueNet though.)


    @property
    def device(self):
        """
        What device is this network on?
        :return:
        """
        return next(self.parameters()).device


    def _do_flat(self, x):
        """
        Flatten out the inputs as required for input to the network.
        :param x:
        :return:
        """
        # If we are using flat vectors, then make sure we flatten out the dimensions.
        if self.type == 'flat':
            x = x.reshape((x.shape[0], -1))
        else:
            # If we are using images, we need to sort the dimensions out a bit.
            x = x.permute(1, 4, 0, 2, 3)
            x = ch.cat([_o for _o in x], dim=0)
            x = x.permute(1, 0, 2, 3)
        return x


    def forward(self, x):
        """
        Call the network.
        :param x: (tensor): input to the policy network.
        :return: (tensor): logits of the actions.
        """
        x = x.type(next(self.parameters()).type())
        x = self._do_flat(x)
        x = self.trunk(x)
        x = x - ch.log(ch.sum(x.exp(), dim=-1)).unsqueeze(-1)   # Normalize the vector so that it is actual log probs.
        return x


    def get_loglikelihood(self, lp, actions):
        """
        Compute the log probability of the actions.
        :param lp:          (tensor):   distribution over actions.
        :param actions:     (tensor):   actions.
        :return:            (tensor):   log probabilities of actions.
        """
        try:
            if type(lp) == tuple:
                lp = lp[0]
            p = F.softmax(lp, dim=-1)
            dist = ch.distributions.categorical.Categorical(p)
            if (actions.size()[-1] > 1) and (actions.ndim > 1):
                actions = actions[:,0]
            return dist.log_prob(actions.long())
        except Exception as e:
            print(e)
            raise ValueError("Numerical error")


    def sample_action(self, states, _eval=False, return_dist=False):
        """
        Evaluate the policy network, and then choose an action.
        :param states:      (tensor):   the states at which to evaluate.
        :param _eval:       (bool):     take the highest probability action, or sample from dist.
        :return:            (tensor):   tensor of actions.
        """
        probs = self.forward(states)
        action = self.sample(probs, _eval=_eval).long()

        if return_dist:
            return action, probs
        else:
            return action


    def sample(self, log_probs, _eval=False):
        """
        Sample an action from the action distribution.
        :param log_probs:   (tensor):   log probabilities of actions.
        :param _eval:       (bool):     take the highest probability action, or sample from dist.
        :return:            (tensor):   tensor of actions.
        """
        if type(log_probs) == tuple:
            log_probs = log_probs[0]
        probs = F.softmax(log_probs, dim=-1)
        if not _eval:
            dist = ch.distributions.categorical.Categorical(probs)
            actions = dist.sample()
        else:
            if type(probs) == tuple:
                probs = probs[0]
            if len(probs.shape) > 1:
                # Correctly formed...
                actions = ch.argmax(probs, dim=1)
            else:
                # Riskily formed...
                actions = ch.argmax(probs).unsqueeze(0)
        return actions.long()


    def density(self, states, actions):
        """
        Evaluate the density of the specified actions at the specified states.
        :param states:      (tensor):   states at which to evaluate.
        :param actions:     (tensor):   actions to evaluate probability of.
        :return:            (tensor)    log probability of actions.
        """
        p, _ = self.forward(states)
        return self.get_loglikelihood(p, actions)


    def calc_kl(self, p, q, get_mean=True):  # TODO: does not return a list
        p, q = p.squeeze(), q.squeeze()
        assert ch.all(p > 0), 'Must supply probabilities.'
        assert ch.all(q > 0), 'Must supply probabilities.'
        assert shape_equal_cmp(p, q)
        kl = (p * (ch.log(p) - ch.log(q))).sum(-1)
        return kl

    def entropies(self, p):
        entropies = (p * ch.log(p)).sum(dim=1)
        return entropies


# This is some slightly strange code, but it allows us to explicitly
# link each of the networks above to a string that we can use to retrieve
# the correct network.
POLICY_NETS = {"DiscPolicy": DiscPolicy, }
VALUE_NETS = {"ValueNet": ValueDenseNet, }

def policy_net_with_name(name):
    return POLICY_NETS[name]

def value_net_with_name(name):
    return VALUE_NETS[name]
