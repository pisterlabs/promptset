"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    Definition of parametrized distributions. Adapted from openai/baselines
"""
import numpy as np
import tensorflow as tf

from typing import Tuple, Callable
from functools import wraps


def expand_dims(func: Callable):
    """ Returns a tensor with a length 1 axis inserted at index 0. """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        result = tf.expand_dims(result, axis=-1)
        return result

    return wrapper

class DiagGaussian(object):
    """ Creates a diagonal Gaussian distribution. """

    def __init__(
            self, 
            ndim: int, 
            action_mean: tf.Tensor,
            action_scale: tf.Tensor
        ):
        """
            Arguments:
            ----------
                ndim: int
                    The dimenstion of actions

                mean: tensorflow.Tensor 
                    Mean stacked on the last axis
                    
                logstd: tensorflow.Tensor 
                    Logstd stacked on the last axis
        """
        self.ndim = ndim
        self.action_mean = action_mean
        self.action_scale = action_scale

    def set_param(self, mean: tf.Tensor, logstd: tf.Tensor):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.math.exp(self.logstd)

    def get_param(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.mean, self.logstd

    def sample(self) -> tf.Tensor:
        """ Get actions in stochastic manner. """
        return self.mean + self.std * np.random.normal(0, 1, np.shape(self.mean))

    def greedy_sample(self) -> tf.Tensor:
        """ Get actions greedily/deterministically. """
        return self.mean

    @expand_dims
    def logp(self, x: tf.Tensor) -> tf.Tensor:
        """ Gets the logarithm of the probability of performing an action. """
        # Normalize the action to the range [-1, 1]
        x = (x - self.action_mean) / self.action_scale

        A = 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1)
        B = 0.5 * np.log(2.0 * np.pi) * float(self.ndim)
        C = tf.reduce_sum(self.logstd, axis=-1)
        y =  A + B + C

        return -y

    @expand_dims
    def kl(self, mean: tf.Tensor, logstd: tf.Tensor) -> tf.Tensor:
        """ Compute the KL Divergence """
        A = logstd - self.logstd
        B = (tf.square(self.std) + tf.square(self.mean - mean))
        C = (2.0 * tf.square(tf.math.exp(logstd)))

        return tf.reduce_sum(A + B / C - 0.5, axis=-1)

    @expand_dims
    def entropy(self):
        """ Gets the distribution entropy. """
        return tf.reduce_sum(
            self.logstd + 0.5 * np.log(2.0 * np.pi * np.e), 
            axis=-1
        )
