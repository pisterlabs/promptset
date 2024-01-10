import tensorflow as tf
import numpy as np
from tensorflow.contrib.distributions import percentile


class SVGD(object):
    def __init__(self, grads_list, vars_list, make_gradient_optimizer, median_heuristic=True):
        self.grads_list = grads_list
        self.vars_list = vars_list
        self.make_gradient_optimizer = make_gradient_optimizer
        self.num_particles = len(vars_list)
        self.median_heuristic = median_heuristic
        self.update_op = self.build_optimizer()

    @staticmethod
    def svgd_kernel(flatvars_list, median_heuristic=True):
        # For pairwise distance in a matrix form, I use the following reference:
        #       https://stackoverflow.com/questions/37009647
        #               /compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
        stacked_vars = tf.stack(flatvars_list)
        norm = tf.reduce_sum(stacked_vars*stacked_vars, 1)
        norm = tf.reshape(norm, [-1, 1])
        pairwise_dists = norm - 2 * tf.matmul(stacked_vars, tf.transpose(stacked_vars)) + tf.transpose(norm)

        # For median in TensorFlow, I use the following reference:
        #       https://stackoverflow.com/questions/43824665/tensorflow-median-value
        def _percentile(x, interpolation):
            return percentile(x, 50.0, interpolation=interpolation)

        if median_heuristic:
            median = (_percentile(pairwise_dists, 'lower') + _percentile(pairwise_dists, 'higher')) / 2.
            median = tf.cast(median, tf.float32)
            h = tf.sqrt(0.5 * median / tf.log(len(flatvars_list) + 1.))

        if len(flatvars_list) == 1:
            h = 1.

        # kernel computation
        Kxy = tf.exp(- pairwise_dists / h ** 2 / 2)
        dxkxy = - tf.matmul(Kxy, stacked_vars)
        sumkxy = tf.reduce_sum(Kxy, axis=1, keep_dims=True)
        dxkxy = (dxkxy + stacked_vars * sumkxy) / tf.pow(h, 2)
        return Kxy, dxkxy

    def build_optimizer(self):
        flatgrads_list, flatvars_list = [], []

        for grads, vars in zip(self.grads_list, self.vars_list):
            flatgrads, flatvars = self.flatten_grads_and_vars(grads, vars)
            flatgrads_list.append(flatgrads)
            flatvars_list.append(flatvars)

        # gradients of SVGD
        Kxy, dxkxy = self.svgd_kernel(flatvars_list, self.median_heuristic)
        stacked_grads = tf.stack(flatgrads_list)
        stacked_grads = (tf.matmul(Kxy, stacked_grads) + dxkxy) / self.num_particles
        flatgrads_list = tf.unstack(stacked_grads, self.num_particles)

        # make gradients for each particle
        grads_list = []
        for flatgrads, vars in zip(flatgrads_list, self.vars_list):
            start = 0
            grads = []
            for var in vars:
                shape = self.var_shape(var)
                size = int(np.prod(shape))
                end = start + size
                grads.append(tf.reshape(flatgrads[start:end], shape))
                start = end
            grads_list.append(grads)

        # optimizer
        update_ops = []
        for grads, vars in zip(grads_list, self.vars_list):
            opt = self.make_gradient_optimizer()
            # gradient ascent
            update_ops.append(opt.apply_gradients([(-g, v) for g, v in zip(grads, vars)]))
        return tf.group(*update_ops)

    def flatten_grads_and_vars(self, grads, vars):
        """Flatten gradients and variables (from openai/baselines/common/tf_util.py)

        :param grads: list of gradients
        :param vars: list of variables
        :return: two lists of flattened gradients and varaibles
        """
        flatgrads =  tf.concat(axis=0, values=[
            tf.reshape(grad if grad is not None else tf.zeros_like(var), [self.num_elements(var)])
            for (var, grad) in zip(vars, grads)])
        flatvars = tf.concat(axis=0, values=[
            tf.reshape(var, [self.num_elements(var)])
            for var in vars])
        return flatgrads, flatvars

    def num_elements(self, var):
        return int(np.prod(self.var_shape(var)))

    @staticmethod
    def var_shape(var):
        out = var.get_shape().as_list()
        assert all(isinstance(a, int) for a in out), \
            'shape function assumes that shape is fully known'
        return out


class Ensemble(object):
    def __init__(self, grads_list, vars_list, make_gradient_optimizer):
        self.grads_list = grads_list
        self.vars_list = vars_list
        self.make_gradient_optimizer = make_gradient_optimizer
        self.num_particles = len(vars_list)
        self.update_op = self.build_optimizer()

    def build_optimizer(self):
        # optimizer
        update_ops = []
        for grads, vars in zip(self.grads_list, self.vars_list):
            opt = self.make_gradient_optimizer()
            # gradient ascent
            update_ops.append(opt.apply_gradients([(-g, v) for g, v in zip(grads, vars)]))
        return tf.group(*update_ops)


class AdagradOptimizer(object):
    def __init__(self, learning_rate=1e-3, alpha=0.9, fudge_factor=1e-6):
        self.learning_rate = tf.constant(learning_rate)
        self.alpha = alpha
        self.fudge_factor = tf.constant(fudge_factor)

    def apply_gradients(self, gvs):
        v_update_ops = []
        for gv in gvs:
            g, v = gv
            historical_grad = tf.Variable(tf.zeros_like(g), trainable=False)
            alpha = tf.Variable(0.0, trainable=False)
            historical_grad_update_op = historical_grad.assign(alpha*historical_grad+(1.-alpha)*g**2)
            with tf.control_dependencies([historical_grad_update_op]):
                adj_grad = tf.div(g, self.fudge_factor + tf.sqrt(historical_grad))
                v_update_op = v.assign(v - self.learning_rate * adj_grad)
                with tf.control_dependencies([v_update_op]):
                    alpha_update_op = alpha.assign(self.alpha)
            v_update_ops.append(alpha_update_op)

        return tf.group(*v_update_ops)


