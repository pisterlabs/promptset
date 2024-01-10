"""
This module defines the REINFORCE agent.
This agent consists of a graph neural network and a policy network, and is trained through REINFORCE algorithm.
Implemented with tensorflow 1.15.
"""
import numpy as np
import bisect
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from params import args
from algo.agent import Agent
from spark_env.job import Job
from spark_env.stage import Stage
from algo.learn import graph_nn
from algo.learn.msg_passing import MsgPassing


class ReinforceAgent(Agent):
    def __init__(self, sess, stage_input_dim, job_input_dim, hidden_dims, output_dim, max_depth, executor_levels,
                 activate_fn, eps, optimizer=tf.train.AdamOptimizer, scope='reinforce_agent'):
        super(ReinforceAgent, self).__init__()
        # ================ 1. basic properties ================
        self.sess = sess
        self.stage_input_dim = stage_input_dim
        self.job_input_dim = job_input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.executor_levels = executor_levels
        self.activate_fn = activate_fn
        self.eps = eps
        self.optimizer = optimizer
        self.scope = scope

        # ================ 2. msg passing, GNNs (the left side of the reinforce agent), and validation masks ================
        self.msg_passing = MsgPassing()
        self.stage_inputs = tf.placeholder(tf.float32, [None, self.stage_input_dim])  # dim 0 = total_num_stages or batch_size?
        self.job_inputs = tf.placeholder(tf.float32, [None, self.job_input_dim])      # dim 0 = total_num_jobs or batch_size?

        self.gcn = graph_nn.GraphCNN(
            self.stage_inputs, self.stage_input_dim, self.hidden_dims, self.output_dim,
            self.max_depth, self.activate_fn, self.scope
        )
        self.gsn = graph_nn.GraphSNN(
            tf.concat([self.stage_inputs, self.gcn.outputs], axis=1),      # x_v^i and e_v^i are inputs of DAG summary
            self.stage_input_dim + self.output_dim,                        # dim(x_v^i) + dim(e_v^i)
            self.hidden_dims, self.output_dim, self.activate_fn, self.scope
        )

        self.stage_valid_mask = tf.placeholder(tf.float32, [None, None])        # (batch_size, total_num_stages)
        self.job_valid_mask = tf.placeholder(tf.float32, [None, None])          # (batch_size, num_jobs * num_exec_limits)
        self.job_summ_backward_map = tf.placeholder(tf.float32, [None, None])   # (total_num_stages, num_jobs)

        # the following two vars are input from current state to indicate whether the stage is runnable and the exec_limit is legal
        self.stage_act_vec = tf.placeholder(tf.float32, [None, None])           # (batch_size, total_num_stages)
        self.job_act_vec = tf.placeholder(tf.float32, [None, None, None])       # (batch_size, total_num_jobs, num_limits)

        # ================ 3. claim problem-specific operations (to get the loss) ===========
        # self.stage_act_probs (batch_size, total_num_stages): the prob. distribution of next stage
        # self.job_act_probs (batch_size, total_num_jobs, num_limits): the prob. distribution of exec_limit for each job
        self.stage_act_probs, self.job_act_probs = self.policy_network(
            self.stage_inputs, self.gcn.outputs, self.job_inputs, self.gsn.summaries[0], self.gsn.summaries[1],
            self.stage_valid_mask, self.job_valid_mask, self.job_summ_backward_map, self.activate_fn
        )

        # the chosen next stage, of shape (batch_size, 1)
        logits = tf.log(self.stage_act_probs)
        # why add noise? It's a trick learned from OpenAI's implementation, below is the explanation:
        # note that the vanilla use of the policy gradient family (e.g., A2C) is on-policy (has to use data sampled from the current policy).
        # Epsilon-greedy creates a bias in the data (because sometimes the action is sampled from random, not just from the current policy).
        # You will need a correction, such as importance sampling, to make the training data unbiased for policy gradient.
        # To avoid all this complication, the standard way of exploration for policy gradient is by increasing the entropy of action distribution
        # and let the random sampling naturally explore
        noise = tf.random_uniform(tf.shape(logits))
        self.stage_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 1)

        # the exec_limit of each job, of shape (batch_size, total_num_jobs, 1)
        logits = tf.log(self.job_act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.job_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 2)

        # stage selected action probability, of shape (batch_size, 1)
        self.selected_stage_prob = tf.reduce_sum(
            tf.multiply(self.stage_act_probs, self.stage_act_vec),
            reduction_indices=1,
            keep_dims=True
        )

        # job selected action probability, of shape (batch_size, total_num_jobs)
        self.selected_job_prob = tf.reduce_sum(
            tf.reduce_sum(tf.multiply(self.job_act_probs, self.job_act_vec), reduction_indices=2),
            reduction_indices=1,
            keep_dims=True
        )

        # advantage term from Monte Carlo or critic, of shape (batch_size, 1)
        self.adv = tf.placeholder(tf.float32, [None, 1])

        # actor loss
        self.adv_loss = tf.reduce_sum(tf.multiply(
            tf.log(self.selected_stage_prob * self.selected_job_prob + self.eps),
            -self.adv
        ))

        # stage entropy
        self.stage_entropy = tf.reduce_sum(tf.multiply(
            self.stage_act_probs,
            tf.log(self.stage_act_probs + self.eps)
        ))

        # prob on each job
        self.prob_on_each_job = tf.reshape(
            tf.sparse_tensor_dense_matmul(self.gsn.summ_mats[0], tf.reshape(self.stage_act_probs, [-1, 1])),
            [tf.shape(self.stage_act_probs)[0], -1]
        )

        # job entropy
        self.job_entropy = tf.reduce_sum(tf.multiply(
            self.prob_on_each_job,
            tf.reduce_sum(tf.multiply(self.job_act_probs, tf.log(self.job_act_probs + self.eps)), reduction_indices=2)
        ))

        # normalized entropy loss over batch size
        self.entropy_loss = self.stage_entropy + self.job_entropy
        self.entropy_loss /= tf.log(tf.cast(tf.shape(self.stage_act_probs)[1], tf.float32)) + tf.log(float(len(self.executor_levels)))

        # use entropy to promote exploration (decay over time)
        self.entropy_weight = tf.placeholder(tf.float32, ())

        # total loss
        self.total_loss = self.adv_loss + self.entropy_weight * self.entropy_loss

        # ================ 4. claim params, gradients, optimizer, and model saver ================
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope
        )
        self.input_params = []
        for param in self.params:
            self.input_params.append(tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_params_op = []
        for idx, param in enumerate(self.input_params):
            self.set_params_op.append(self.params[idx].assign(param))

        self.act_gradients = tf.gradients(self.total_loss, self.params)
        self.lr = tf.placeholder(tf.float32, shape=[])
        self.act_opt = self.optimizer(self.lr).minimize(self.total_loss)
        self.apply_grads = self.optimizer(self.lr).apply_gradients(zip(self.act_gradients, self.params))
        self.model_saver = tf.train.Saver(max_to_keep=args.num_saved_models)

        # ================ 5. param init (load from saved model in default) ================
        self.sess.run(tf.global_variables_initializer())
        if args.saved_model is not None:
            self.model_saver.restore(self.sess, args.saved_model)

    def policy_network(self, stage_inputs, gcn_outputs, job_inputs, gsn_job_summary, gsn_global_summary,
                       stage_valid_mask, job_valid_mask, gsn_summ_backward_map, activate_fn):
        """
        This method implements the right side of the reinforce agent:
        With (1) the raw (stage and job) inputs, (2) DAG level summary, and (3) global level summary,
        use neural networks q and w to get the actions:
            (1) stage selection prob. distribution, and
            (2) parallelism limit on each job.

        Thus, the main part of this method is the two NNs, q and w.
        """
        batch_size = tf.shape(stage_valid_mask)[0]

        # reshape to batch format
        stage_inputs_reshape = tf.reshape(stage_inputs, [batch_size, -1, self.stage_input_dim])
        job_inputs_reshape = tf.reshape(job_inputs, [batch_size, -1, self.job_input_dim])
        gcn_outputs_reshape = tf.reshape(gcn_outputs, [batch_size, -1, self.output_dim])

        # reshape job_summary and global_summary to batch format
        gsn_job_summ_reshape = tf.reshape(gsn_job_summary, [batch_size, -1, self.output_dim])
        gsn_summ_backward_map_extend = tf.tile(tf.expand_dims(gsn_summ_backward_map, axis=0), [batch_size, 1, 1])
        gsn_job_summ_extend = tf.matmul(gsn_summ_backward_map_extend, gsn_job_summ_reshape)

        gsn_global_summ_reshape = tf.reshape(gsn_global_summary, [batch_size, -1, self.output_dim])
        gsn_global_summ_extend_job = tf.tile(gsn_global_summ_reshape, [1, tf.shape(gsn_job_summ_reshape)[1], 1])
        gsn_global_summ_extend_stage = tf.tile(gsn_global_summ_reshape, [1, tf.shape(gsn_job_summ_extend)[1], 1])

        with tf.variable_scope(self.scope):
            # part 1: the probability distribution over stage selection
            merge_stage = tf.concat(
                [stage_inputs_reshape, gcn_outputs_reshape, gsn_job_summ_extend, gsn_global_summ_extend_stage],
                axis=2
            )
            stage_hidden0 = tf_layers.fully_connected(merge_stage, 32, activation_fn=activate_fn)
            stage_hidden1 = tf_layers.fully_connected(stage_hidden0, 16, activation_fn=activate_fn)
            stage_hidden2 = tf_layers.fully_connected(stage_hidden1, 8, activation_fn=activate_fn)
            stage_outputs = tf_layers.fully_connected(stage_hidden2, 1, activation_fn=None)
            stage_outputs = tf.reshape(stage_outputs, [batch_size, -1])    # (batch_size, total_num_stages)

            stage_valid_mask = (stage_valid_mask - 1) * 10000.    # to make those stages which cannot be chosen have very low prob
            stage_outputs = stage_outputs + stage_valid_mask
            stage_outputs = tf.nn.softmax(stage_outputs, dim=-1)

            # part 2: the probability distribution over executor limits
            merge_job = tf.concat(
                [job_inputs_reshape, gsn_job_summ_reshape, gsn_global_summ_extend_job],
                axis=2
            )
            expanded_state = expand_act_on_state(merge_job, [lvl / 50. for lvl in self.executor_levels])
            job_hidden0 = tf_layers.fully_connected(expanded_state, 32, activation_fn=activate_fn)
            job_hidden1 = tf_layers.fully_connected(job_hidden0, 16, activation_fn=activate_fn)
            job_hidden2 = tf_layers.fully_connected(job_hidden1, 8, activation_fn=activate_fn)
            job_outputs = tf_layers.fully_connected(job_hidden2, 1, activation_fn=None)
            job_outputs = tf.reshape(job_outputs, [batch_size, -1])  # (batch_size, num_jobs * num_exec_limits)

            job_valid_mask = (job_valid_mask - 1) * 10000.
            job_outputs = job_outputs + job_valid_mask
            # reshape to (batch_size, num_jobs, num_exec_limits)
            job_outputs = tf.reshape(job_outputs, [batch_size, -1, len(self.executor_levels)])
            job_outputs = tf.nn.softmax(job_outputs, dim=-1)

            return stage_outputs, job_outputs

    def invoke_model(self, obs):
        stage_inputs, job_inputs, jobs, src_job, num_src_exec, frontier_stages, exec_limits, \
            exec_commit, moving_executors, exec_map, action_map = self.translate_state(obs)

        # get msg passing path (with cache)
        gcn_mats, gcn_masks, job_summ_backward_map, running_jobs_mat, jobs_changed = self.msg_passing.get_msg_path(jobs)
        # get valid masks
        stage_valid_mask, job_valid_mask = self.get_valid_masks(jobs, frontier_stages, src_job, num_src_exec, exec_map, action_map)
        # get summ path which ignores the finished stages
        summ_mats = self.get_unfinished_stages_summ_mat(jobs)

        # invoke learning model
        stage_act_probs, job_act_probs, stage_acts, job_acts = self.predict(stage_inputs, job_inputs, stage_valid_mask,
                                                                            job_valid_mask, gcn_mats, gcn_masks, summ_mats,
                                                                            running_jobs_mat, job_summ_backward_map)

        return stage_acts, job_acts, stage_act_probs, job_act_probs, stage_inputs, job_inputs, stage_valid_mask, \
            job_valid_mask, gcn_mats, gcn_masks, summ_mats, running_jobs_mat, job_summ_backward_map, exec_map, jobs_changed

    # the following 4 funcs are called in invoke_model()
    def translate_state(self, obs):
        """
        Translate the observation from Schedule.observe() into tf tensor format.
        This func gives the design of raw (feature) input.
        """
        jobs, src_job, num_src_exec, frontier_stages, exec_limits, exec_commit, moving_executors, action_map = obs
        total_num_stages = int(np.sum(job.num_stages for job in jobs))

        # set stage_inputs and job_inputs
        stage_inputs = np.zeros([total_num_stages, self.stage_input_dim])
        job_inputs = np.zeros([len(jobs), self.job_input_dim])

        exec_map = {}         # {job: num of executors allocated to it}
        for job in jobs:
            exec_map[job] = len(job.executors)
        # count the moving executors in
        for stage in moving_executors.moving_executors.values():
            exec_map[stage.job] += 1
        # count exec_commit in
        for src in exec_commit.commit:
            job = None
            if isinstance(src, Job):
                job = src
            elif isinstance(src, Stage):
                job = src.job
            elif src is None:
                job = None
            else:
                print('Source', src, 'unknown!')
                exit(1)
            for stage in exec_commit.commit[src]:
                if stage is not None and stage.job != job:
                    exec_map[stage.job] += exec_commit.commit[src][stage]

        # gather job level inputs (thw following demonstrates the raw feature design)
        job_idx = 0
        for job in jobs:
            job_inputs[job_idx, 0] = exec_map[job] / 20.                   # dim0: num executors
            job_inputs[job_idx, 1] = 2 if job is src_job else -2           # dim1: cur exec belongs to this job or not
            job_inputs[job_idx, 2] = num_src_exec / 20.                    # dim2: num of src execs
            job_idx += 1
        # gather stage level inputs
        stage_idx = 0
        job_idx = 0
        for job in jobs:
            for stage in job.stages:
                stage_inputs[stage_idx, :3] = job_inputs[job_idx, :3]
                stage_inputs[stage_idx, 3] = (stage.num_tasks - stage.next_task_idx) * stage.tasks[-1].duration / 100000.  # remaining task execution tm
                stage_inputs[stage_idx, 4] = (stage.num_tasks - stage.next_task_idx) / 200.                                # num of remaining tasks
                stage_idx += 1
            job_idx += 1

        return stage_inputs, job_inputs, jobs, src_job, num_src_exec, frontier_stages, exec_limits, \
            exec_commit, moving_executors, exec_map, action_map

    def get_valid_masks(self, jobs, frontier_stages, src_job, num_src_exec, exec_map, action_map):
        job_valid_mask = np.zeros([1, len(jobs) * len(self.executor_levels)])
        job_valid = {}       # {job: True or False}

        base = 0
        for job in jobs:
            # new executor level depends on the src exec
            if job is src_job:
                # + 1 because we want at least one exec for this job
                least_exec_amount = exec_map[job] - num_src_exec + 1
            else:
                least_exec_amount = exec_map[job] + 1
            assert 0 < least_exec_amount <= self.executor_levels[-1] + 1

            # find the idx of the first valid executor limit
            exec_level_idx = bisect.bisect_left(self.executor_levels, least_exec_amount)
            if exec_level_idx >= len(self.executor_levels):
                job_valid[job] = False
            else:
                job_valid[job] = True

            # jobs behind exec_level_idx are valid
            for lvl in range(exec_level_idx, len(self.executor_levels)):
                job_valid_mask[0, base + lvl] = 1
            base += self.executor_levels[-1]

        total_num_stages = int(np.sum(job.num_stages for job in jobs))
        stage_valid_mask = np.zeros([1, total_num_stages])
        for stage in frontier_stages:
            if job_valid[stage.job]:
                act = action_map.inverse_map[stage]
                stage_valid_mask[0, act] = 1

        return stage_valid_mask, job_valid_mask

    @staticmethod
    def get_unfinished_stages_summ_mat(jobs):
        """
        Add a connection from the unfinished stages to the summarized node.
        """
        total_num_stages = np.sum([job.num_stages for job in jobs])
        summ_row_idx, summ_col_idx, summ_data = [], [], []
        summ_shape = (len(jobs), total_num_stages)

        base = 0
        job_idx = 0
        for job in jobs:
            for stage in job.stages:
                if not stage.all_tasks_done:
                    summ_row_idx.append(job_idx)
                    summ_col_idx.append(base + stage.idx)
                    summ_data.append(1)

            base += job.num_stages
            job_idx += 1

        return tf.SparseTensorValue(
            indices=np.mat([summ_row_idx, summ_col_idx]).transpose(),
            values=summ_data,
            dense_shape=summ_shape
        )

    def predict(self, stage_inputs, job_inputs, stage_valid_mask, job_valid_mask, gcn_mats, gcn_masks, summ_mats,
                running_jobs_mat, job_summ_backward_map):
        return self.sess.run(
            [self.stage_act_probs, self.job_act_probs, self.stage_acts, self.job_acts],
            feed_dict={
                i: d for i, d in zip(
                    [self.stage_inputs] + [self.job_inputs] + [self.stage_valid_mask] + [self.job_valid_mask] +
                    self.gcn.adj_mats + self.gcn.masks + self.gsn.summ_mats + [self.job_summ_backward_map],

                    [stage_inputs] + [job_inputs] + [stage_valid_mask] + [job_valid_mask] +
                    gcn_mats + gcn_masks + [summ_mats, running_jobs_mat] + [job_summ_backward_map]
                )
            }
        )

    # the following 4 funcs (return sess.run()) are called in model training
    def apply_gradients(self, gradients, lr):
        self.sess.run(
            self.apply_grads,
            feed_dict={
                i: d for i, d in zip(self.act_gradients + [self.lr], gradients + [lr])
            }
        )

    def gcn_forward(self, stage_inputs, summ_mats):
        return self.sess.run(
            [self.gsn.summaries],
            feed_dict={
                i: d for i, d in zip([self.stage_inputs] + self.gsn.summ_mats, [stage_inputs] + summ_mats)
            }
        )

    def get_gradients(self, stage_inputs, job_inputs, stage_valid_mask, job_valid_mask, gcn_mats, gcn_masks, summ_mats,
                      running_jobs_mat, job_summ_backward_map, stage_act_vec, job_act_vec, adv, entropy_weight):
        return self.sess.run(
            [self.act_gradients, [self.adv_loss, self.entropy_loss]],
            feed_dict={
                i: d for i, d in zip(
                    [self.stage_inputs] + [self.job_inputs] + [self.stage_valid_mask] + [self.job_valid_mask] +
                    self.gcn.adj_mats + self.gcn.masks + self.gsn.summ_mats + [self.job_summ_backward_map] +
                    [self.stage_act_vec] + [self.job_act_vec] + [self.adv] + [self.entropy_weight],

                    [stage_inputs] + [job_inputs] + [stage_valid_mask] + [job_valid_mask] +
                    gcn_mats + gcn_masks + [summ_mats, running_jobs_mat] + [job_summ_backward_map] +
                    [stage_act_vec] + [job_act_vec] + [adv] + [entropy_weight]
                )
            }
        )

    def set_params(self, input_params):
        self.sess.run(
            self.set_params_op,
            feed_dict={
                i: d for i, d in zip(self.input_params, input_params)
            }
        )

    def get_action(self, obs):
        """
        Get the next-to-schedule stage and the exec limits to it by parsing the output of the agent.
        """
        jobs, src_job, num_src_exec, frontier_stages, exec_limits, exec_commit, moving_executors, action_map = obs
        if len(frontier_stages) == 0:      # no act
            return None, num_src_exec
        stage_acts, job_acts, stage_act_probs, job_act_probs, stage_inputs, job_inputs, stage_valid_mask, job_valid_mask, \
            gcn_mats, gcn_masks, summ_mats, running_jobs_mat, job_summ_backward_map, exec_map, jobs_changed = self.invoke_model(obs)
        if sum(stage_valid_mask[0, :]) == 0:        # no valid stage to assign
            return None, num_src_exec

        assert stage_valid_mask[0, stage_acts[0]] == 1
        # parse stage action, get the to-be-scheduled stage
        stage = action_map[stage_acts[0]]
        # find the corresponding job
        job_idx = jobs.index(stage.job)
        assert job_valid_mask[0, job_acts[0, job_idx] + len(self.executor_levels) * job_idx] == 1
        # parse exec limit action
        if stage.job is src_job:
            agent_exec_act = self.executor_levels[job_acts[0, job_idx]] - exec_map[stage.job] + num_src_exec
        else:
            agent_exec_act = self.executor_levels[job_acts[0, job_idx]] - exec_map[stage.job]
        use_exec = min(
            stage.num_tasks - stage.next_task_idx - exec_commit.stage_commit[stage] - moving_executors.count(stage),
            agent_exec_act,
            num_src_exec
        )

        return stage, use_exec

    def get_params(self):
        return self.sess.run(self.params)

    def save_model(self, save_path):
        return self.sess.run(self.sess, save_path)


def expand_act_on_state(state, sub_acts):
    batch_size = tf.shape(state)[0]
    num_stages = tf.shape(state)[1]
    num_features = state.shape[2].value  # deterministic
    expand_dim = len(sub_acts)

    # replicate the state
    state = tf.tile(state, [1, 1, expand_dim])
    state = tf.reshape(state, [batch_size, num_stages * expand_dim, num_features])

    # prepare the appended sub-actions
    sub_acts = tf.constant(sub_acts, dtype=tf.float32)
    sub_acts = tf.reshape(sub_acts, [1, 1, expand_dim])
    sub_acts = tf.tile(sub_acts, [1, 1, num_stages])
    sub_acts = tf.reshape(sub_acts, [1, num_stages * expand_dim, 1])
    sub_acts = tf.tile(sub_acts, [batch_size, 1, 1])      # now the first two dim of sub_acts are as the same as state's

    # concatenate
    concat_state = tf.concat([state, sub_acts], axis=2)   # dim2 = num_features + 1
    return concat_state


def leaky_relu(features, alpha=0.2, name=None):
    """
    Implement the leaky ReLU activate function.
    f(x) = x if x > 0 else alpha * x.
    """
    with ops.name_scope(name, 'LeakyReLU', [features, alpha]):
        features = ops.convert_to_tensor(features, name='features')
        alpha = ops.convert_to_tensor(alpha, name='alpha')
        return math_ops.maximum(alpha * features, features)
