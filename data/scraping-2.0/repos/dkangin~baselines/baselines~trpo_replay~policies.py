# The code is based on the ACKTR implementation from OpenAI baselines
# This code implements method by Kangin & Pugeault "On-Policy Trust Region Policy Optimisation with Replay Buffers"
import numpy as np
import tensorflow as tf
from baselines.trpo_replay.utils import dense, kl_div
import baselines.common.tf_util as U
from baselines.trpo_replay import kfac

class GaussianMlpPolicy(object):
    def __init__(self, ob_dim, ac_dim, name='pi'):
        self.name = name
        self.keep_prob = tf.placeholder (tf.float32, shape=())
        ob_no = tf.placeholder(tf.float32, shape=[None, ob_dim*2], name="ob") # batch of observations
        oldac_na = tf.placeholder(tf.float32, shape=[None, ac_dim], name="ac") # batch of actions previous actions
        oldac_dist = tf.placeholder(tf.float32, shape=[None, ac_dim*2], name="oldac_dist") # batch of actions previous action distributions
        adv_n = tf.placeholder(tf.float32, shape=[None], name="adv") # advantage function estimate
        adv_new_flag = tf.placeholder (tf.bool, shape = [None], name ='adv_new_flag')
        wd_dict = {}
        h1 = tf.nn.tanh(dense(ob_no, 64, "h1", weight_init=U.normc_initializer(1.0), bias_init=0.0, weight_loss_dict=wd_dict))
        h1 = tf.nn.dropout (h1, self.keep_prob)
        h2 = tf.nn.tanh(dense(h1, 64, "h2", weight_init=U.normc_initializer(1.0), bias_init=0.0, weight_loss_dict=wd_dict))
        h2 = tf.nn.dropout (h2, self.keep_prob)
        mean_na = dense(h2, ac_dim, "mean", weight_init=U.normc_initializer(0.1), bias_init=0.0, weight_loss_dict=wd_dict) # Mean control output
        self.wd_dict = wd_dict
        std_1a = dense(h2, ac_dim, "logstd", weight_init=U.normc_initializer(0.1), bias_init=0.0, weight_loss_dict=wd_dict)#tf.get_variable("logstd", [ac_dim], tf.float32, tf.zeros_initializer()) # Variance on outputs
        #std_1a = tf.expand_dims(logstd_1a, 0)
        std_na = tf.maximum(0.2, tf.minimum(5., tf.multiply(std_1a, std_1a))) 
        #std_na = tf.tile(std_1a, [tf.shape(mean_na)[0], 1])
        ac_dist = tf.concat([tf.reshape(mean_na, [-1, ac_dim]), tf.reshape(std_na, [-1, ac_dim])], 1)
        sampled_ac_na = tf.random_normal(tf.shape(ac_dist[:,ac_dim:])) * ac_dist[:,ac_dim:] + ac_dist[:,:ac_dim] # This is the sampled action we'll perform.
        logprobsampled_n = - tf.reduce_sum(tf.log(ac_dist[:,ac_dim:]), axis=1) - 0.5 * tf.log(2.0*np.pi)*ac_dim - 0.5 * tf.reduce_sum(tf.square(ac_dist[:,:ac_dim] - sampled_ac_na) / (tf.square(ac_dist[:,ac_dim:])), axis=1) # Logprob of sampled action
        logprob_n = - tf.reduce_sum(tf.log(ac_dist[:,ac_dim:]), axis=1) - 0.5 * tf.log(2.0*np.pi)*ac_dim - 0.5 * tf.reduce_sum(tf.square(ac_dist[:,:ac_dim] - oldac_na) / (tf.square(ac_dist[:,ac_dim:])), axis=1) # Logprob of previous actions under CURRENT policy (whereas oldlogprob_n is under OLD policy)
        kl = tf.reduce_mean(kl_div(tf.gather(oldac_dist, tf.reshape(tf.where(adv_new_flag), [-1]), axis=0), tf.gather(ac_dist, tf.reshape(tf.where(adv_new_flag), [-1]), axis=0), ac_dim))
        #kl = .5 * tf.reduce_mean(tf.square(logprob_n - oldlogprob_n)) # Approximation of KL divergence between old policy used to generate actions, and new policy used to compute logprob_n
        #kl = tf.reduce_mean(kl_div(oldac_dist, ac_dist, ac_dim))
        surr = - tf.reduce_mean(adv_n * logprob_n) # Loss function that we'll differentiate to get the policy gradient
        self.surr_original = surr
        surr = surr + 100 * tf.maximum (0., kl - 0.1)
        surr_sampled = - tf.reduce_mean(logprob_n) # Sampled loss of the policy
        self.surr_sampled = surr_sampled
        self._act = U.function([ob_no, self.keep_prob], [sampled_ac_na, ac_dist, logprobsampled_n]) # Generate a new action and its logprob
        #self.compute_kl = U.function([ob_no, oldac_na, oldlogprob_n], kl) # Compute (approximate) KL divergence between old policy and new policy
        self.compute_kl = U.function([ob_no, oldac_dist, self.keep_prob, adv_new_flag], kl)
        self.ob_no = ob_no
        self.oldac_na = oldac_na
        self.adv_n = adv_n
        self.oldac_dist = oldac_dist

        weights_size = 0
        vf_var_list = []
        var_shape = []
        var_size = []
        for var in tf.trainable_variables():
            if "pi" in var.name:
                vf_var_list.append(var)
                var_shape.append(tf.shape(var).eval())
                var_size.append(tf.size(var).eval())
                weights_size += tf.size(var).eval()
        print ('weights_size: ', weights_size)
        self.var_shape = var_shape
        self.var_size = var_size
        self.weights = tf.placeholder(tf.float32, shape=[weights_size], name='curr_weights')
        i = 0
        j = 0
        self.set_weights_ops = []
        self.get_weights_ops = []

        for var in vf_var_list:
            var_shape = self.var_shape[j]#tf.shape(var).eval()
            var_size = self.var_size[j]#tf.size(var).eval()
            weights_curr = tf.reshape (self.weights[i:i+var_size], var_shape)
            self.set_weights_ops.append(tf.assign (var, weights_curr))
            self.get_weights_ops.append(var)
            i += var_size
            j += 1        
        self.set_weights_ops = tf.group(*self.set_weights_ops)
        self.get_weights_ops = U.function ([], self.get_weights_ops)#tf.group (*self.get_weights_ops)
        self.update_info = ((ob_no, oldac_na, adv_n, adv_new_flag, oldac_dist, self.keep_prob), surr, surr_sampled) # Input and output variables needed for computing loss
        U.initialize() # Initialize uninitialized TF variables

    def create_policy_optimiser (self, stepsize):
        inputs, loss, loss_sampled = self.update_info
        optim = kfac.KfacOptimizer(learning_rate=stepsize, cold_lr=stepsize*(1-0.9), momentum=0.9, kfac_update=2,\
                                   epsilon=1e-2, stats_decay=0.99, async=1, cold_iter=1,
                                   weight_decay_dict=self.wd_dict, max_grad_norm=None)
        
        pi_var_list = []
        for var in tf.trainable_variables():
            if "pi" in var.name:
                pi_var_list.append(var)

        update_op, q_runner = optim.minimize(loss, loss_sampled, var_list=pi_var_list)
        do_update = U.function(inputs, update_op)
        U.initialize()
        return do_update, q_runner    

    def act(self, ob):
        ac, ac_dist, logp = self._act(ob[None], 1.0)
        #print ('ac_dist[0]: ', ac_dist[0])
        return ac[0], ac_dist[0], logp[0]

    def get_trainable_weights (self):
        return tf.get_collection (tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
   
    def flatten (self, result):
        flattened_array = []
        for i in range(len(result)):
            flattened_array.extend(np.reshape (result[i], [-1]))
        return flattened_array

    def get_weights (self):
        result = self.get_weights_ops()
        #print ('get_weights (self): ', result)
        return self.flatten (result)
        #return [item.flatten() for sublist in result for item in sublist]

    def set_weights (self, weights):
        #weights [0] = 0
        self.set_weights_ops.run(feed_dict={self.weights: weights})

    def get_loss (self, ob_no, oldac_na, adv_n, oldac_dist):
        return self.surr_sampled.eval(feed_dict = {self.ob_no: ob_no, self.oldac_na: oldac_na, self.adv_n: adv_n, self.oldac_dist: oldac_dist, self.keep_prob:1.0})

    def get_target_updates(self, vars, target_vars):
        init_updates = []
        assert len(vars) == len(target_vars)
        for var, target_var in zip(vars, target_vars):
            init_updates.append(tf.assign(target_var, var))
        assert len(init_updates) == len(vars)
        return tf.group(*init_updates)

