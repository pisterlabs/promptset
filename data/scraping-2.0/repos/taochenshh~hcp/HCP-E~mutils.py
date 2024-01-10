import gym
import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def online_variance(data):
    '''
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    '''
    n = 0
    mean = M2 = 0.0

    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2

    if n < 2:
        return float('nan')
    else:
        return M2 / n


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def explained_variance(ypred, y):
    """
    *** copy from openai/baselines ***
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def add_weight_decay(nets, weight_decay, skip_list=()):
    decay, no_decay = [], []
    for net in nets:
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if "bias" in name or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)


class OnlineMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, shape):
        self.mean = torch.zeros(shape)
        self.m2 = torch.zeros(shape)
        self.var = torch.zeros(shape)
        self.std = torch.zeros(shape)
        self.count = 0

    def update(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.contiguous().view(-1, self.mean.size()[1]).float()
        x_count = x.size()[0]
        x_mean = torch.mean(x, dim=0)
        x_m2 = (torch.mean(torch.pow(x, 2),
                           dim=0) - torch.pow(x_mean, 2)) * x_count
        delta = x_mean - self.mean
        total_count = self.count + x_count
        self.mean = (self.count * self.mean +
                     x_count * x_mean) / total_count
        self.m2 = self.m2 + x_m2
        self.m2 += torch.pow(delta, 2) * self.count * x_count / total_count
        self.count = total_count
        self.var = self.m2 / self.count
        self.std = torch.sqrt(self.var)


def print_red(skk):
    print("\033[91m {}\033[00m".format(skk))


def print_green(skk):
    print("\033[92m {}\033[00m".format(skk))


def print_yellow(skk):
    print("\033[93m {}\033[00m".format(skk))


def print_blue(skk):
    print("\033[94m {}\033[00m".format(skk))


def print_purple(skk):
    print("\033[95m {}\033[00m".format(skk))


def print_cyan(skk):
    print("\033[96m {}\033[00m".format(skk))
