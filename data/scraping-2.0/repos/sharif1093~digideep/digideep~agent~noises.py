"""
This module is dedicated to noise models used in other methods.

Each noise class should implement the ``__call__`` method. See the examples :class:`EGreedyNoise` and :class:`OrnsteinUhlenbeckNoise`.
"""
import numpy as np

class EGreedyNoise:
    """This class implements simple e-greedy noise. The noise is sampled from uniform distribution.
    
    Args:
        std (float): Standard deviation of the noise.
        e (float): The probability of choosing a noisy action.
        lim (float): Boundary of the noise (noise will be clipped beyond this value.)

    Note:
        This class is not dependant on its history.
    """

    def __init__(self, **params):
        self.params = params
        self.state = {}
    
    def reset(self):
        pass
    
    def __call__(self, action):
        # Add some noise to the action
        std = self.params["std"]
        e   = self.params["e"]
        lim = self.params["lim"] # A scalar
        
        noise = std * lim * np.random.randn(*action.shape)
        action += noise
        action = np.clip(action, a_min=-lim, a_max=lim)

        # By chance e, choose a completely random action.
        # By chance 1-e, let the current noised action survive.
        # We use this beautiful trick from openai to do the above:
        random_action = np.random.uniform(low=-lim, high=lim, size=action.shape)
        
        # When we have a batch of actions, this should be like this:
        # choice = np.random.binomial(1, e, size=(action.shape[0],1))
        # Where action.shape[0] is the batch-size
        choice = np.random.binomial(1, e, size=(action.shape[0],1))
        noise = choice * (random_action - action)
        action += noise
        return action
    
    def state_dict(self):
        return None
    def load_state_dict(self, state_dict):
        pass



class OrnsteinUhlenbeckNoise:
    """An implementation of the `Ornstein-Uhlenbeck noise <https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process>`_.

    The noise model is :math:`{\displaystyle dx_{t}=\theta (\mu -x_{t})\,dt+\sigma \,dW_{t}}`.

    Args:
        mu: Parameter :math:`\mu` which indicates the final value that :math:`x` will converge to.
        theta: Parameter :math:`\theta`.
        sigma: Parameter :math:`\sigma` which is the std of the additional normal noise.
        lim: The action limit, which can be a :obj:`np.array` for a vector of actions.
    
    Note:
        This class is state serializable.
    """

    def __init__(self, **params):
        """ Ornstein-Uhlenbeck process noise generator
        params: mu, theta, sigma
        states: X
        """

        self.params = params
        self.state = {}
        self.state['needs_reset'] = True

    def reset(self, action):
        self.state['X'] = np.ones_like(action) * self.params['mu']
        self.state['needs_reset'] = False

    def __call__(self, action):
        if self.state['needs_reset']:
            self.reset(action)
        dx = self.params['theta'] * (self.params['mu'] - self.state['X'])
        dx = dx + self.params['sigma'] * np.random.randn(*self.state['X'].shape)
        self.state['X'] = self.state['X'] + dx
        noise = self.state['X'] * self.params["lim"]
        # print("noise", noise)
        action += noise
        return action
    
    def state_dict(self):
        return self.state
    def load_state_dict(self, state_dict):
        self.state.update(state_dict)
