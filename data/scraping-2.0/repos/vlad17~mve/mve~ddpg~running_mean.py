"""
Class for keeping running statistics. Copied from OpenAI baselines,
with the MPI dependency removed. OpenAI implementation based on

https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

This is required, but not yet used, for DDPG popart.
"""

import tensorflow as tf
import numpy as np


class RunningMeanStd:
    """
    Contains variables for tracking mean and std.

    Contains attributes:

    * shape - shape of statistics maintained as a tuple
    * mean - TF tensor for mean
    * std - TF tensor for std
    """

    def __init__(self, scope, shape=(), reuse=None):
        epsilon = 1e-6
        with tf.variable_scope(scope, reuse):
            self._sum = tf.get_variable(
                dtype=tf.float64,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="runningsum", trainable=False)
            self._sumsq = tf.get_variable(
                dtype=tf.float64,
                shape=shape,
                initializer=tf.constant_initializer(epsilon),
                name="runningsumsq", trainable=False)
            self._count = tf.get_variable(
                dtype=tf.float64,
                shape=(),
                initializer=tf.constant_initializer(epsilon),
                name="count", trainable=False)
        self.shape = shape

        self.mean = tf.cond(self._count == 0,
                            lambda: tf.constant(0),
                            lambda: tf.to_float(self._sum / self._count))
        self.std = tf.cond(self._count == 0,
                           lambda: tf.constant(1),
                           lambda: tf.sqrt(tf.maximum(
                               tf.to_float(self._sumsq / self._count)
                               - tf.square(self.mean), epsilon)))

        self._newsum_ph = tf.placeholder(tf.float64, shape)
        self._newsumsq_ph = tf.placeholder(tf.float64, shape)
        self._newcount_ph = tf.placeholder(tf.float64, [])
        self._updates = tf.group(
            tf.assign_add(self._sum, self._newsum_ph),
            tf.assign_add(self._sumsq, self._newsumsq_ph),
            tf.assign_add(self._count, self._newcount_ph))

    def update(self, x):
        """
        Given a numpy array of several observations (each of the shape of the
        statistics that this class is tracking), update the statistics
        accordingly.
        """
        x = x.astype('float64')
        tf.get_default_session().run(self._updates, feed_dict={
            self._newsum_ph: x.sum(axis=0),
            self._newsumsq_ph: np.square(x).sum(axis=0),
            self._newcount_ph: len(x)})

    def tf_normalize(self, x):
        """
        Return a TF tensor for x normalized by the statistics in this class.
        """
        return (x - self.mean) / self.std

    def tf_denormalize(self, x):
        """
        Return a TF tensor for x denormalized by the statistics in this class.
        """
        return x * self.std + self.mean
