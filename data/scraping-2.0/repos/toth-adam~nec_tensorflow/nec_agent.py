import logging
import sys
import os
import argparse
from collections import deque, OrderedDict

import numpy as np
from scipy.signal import lfilter

import tensorflow as tf
import tensorflow.contrib.slim as slim
# from tensorflow.python.client import timeline

from lru import LRU
from pyflann import FLANN
from mmh3 import hash128

from replay_memory import ReplayMemory


log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", help="The id of GPU (default 0)")

args = vars(parser.parse_args())

os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu_id"] if args["gpu_id"] else "0"


class NECAgent:
    def __init__(self, action_vector, cpu_only=False, dnd_max_memory=500000, neighbor_number=50,
                 backprop_learning_rate=1e-4, tabular_learning_rate=1e-3, fully_conn_neurons=128,
                 input_shape=(84, 84, 4), kernel_size=((3, 3), (3, 3), (3, 3), (3, 3)), num_outputs=(32, 32, 32, 32),
                 stride=((2, 2), (2, 2), (2, 2), (2, 2)), delta=1e-3, rep_memory_size=1e5, batch_size=32,
                 n_step_horizon=100, discount_factor=0.99, log_save_directory=None, epsilon_decay_bounds=(5000, 25000),
                 optimization_start=1000, ann_rebuild_freq=10):

        # TÖRÖLNI
        self.seen_states_number = 0

        self._cpu_only = cpu_only

        # ----------- HYPERPARAMETERS ----------- #

        self.delta = delta
        self.initial_epsilon = 1
        self.epsilon_decay_bounds = epsilon_decay_bounds

        # Optimizer parameters
        self.adam_learning_rate = backprop_learning_rate
        self.batch_size = batch_size
        self.optimization_start = optimization_start

        # Tabular parameters
        self.tab_alpha = tabular_learning_rate
        self.dnd_max_memory = int(dnd_max_memory)

        # Reinforcement learning parameters
        self.n_step_horizon = n_step_horizon
        self.discount_factor = discount_factor

        # Convolutional layer parameters
        self._input_shape = input_shape
        self.fully_connected_neuron = fully_conn_neurons
        self._kernel_size = kernel_size
        self._stride = stride
        self._num_outputs = num_outputs

        # Environment specific parameters
        self.action_vector = action_vector
        self.number_of_actions = len(action_vector)
        self.frame_stacking_number = input_shape[-1]

        # ANN Search
        self.ann_rebuild_freq = ann_rebuild_freq
        self.neighbor_number = neighbor_number
        self.anns = {k: AnnSearch(neighbor_number, dnd_max_memory, k) for k in action_vector}

        # Replay memory
        self.replay_memory = ReplayMemory(size=rep_memory_size, stack_size=input_shape[-1])

        #AZ LRU az tf_index:state_hash mert az ann_search alapján kell a sorrendet updatelni mert a dict1-ben
        # updatelni kell dict1 az state_hash:tf_index ez ahhoz kell hogy megnezzem hogy benne van-e tehát milyen
        # legyen a tab_update és hogy melyik indexre a DND-ben
        self.tf_index__state_hash = {k: LRU(self.dnd_max_memory) for k in action_vector}
        self.state_hash__tf_index = {k: {} for k in action_vector}

        # Tensorflow Session object
        self.session = self._create_tf_session()

        # Step numbers
        self.global_step = 0
        self.episode_step = 0
        self.episode_number = 0

        # For logging the total loss and windowed average episode reward
        self.create_list_for_total_losses = True
        self.episode_total_reward = 0
        self.windowed_average_total_reward = deque(maxlen=15)

        # ----------- TENSORFLOW GRAPH BUILDING ----------- #

        self.dnd_placeholder_ops = OrderedDict()
        self.dnd_key_gather_ops = OrderedDict()
        self.dnd_value_gather_ops = OrderedDict()
        self.dnd_scatter_update_placeholder_ops = OrderedDict()
        self.dnd_scatter_update_key_ops = OrderedDict()
        self.dnd_scatter_update_value_ops = OrderedDict()
        self.dnd_value_update_placeholder_ops = OrderedDict()
        self.dnd_key_ops, self.dnd_value_ops = OrderedDict(), OrderedDict()

        if self._cpu_only:
            device = "/cpu:0"
        else:
            device = "/device:GPU:0"

        with tf.device(device):
            self.state = tf.placeholder(shape=[None, *self._input_shape], dtype=tf.float32, name="state")
            # Always better to use smaller kernel size! These layers are from OpenAI
            # Learning Atari: An Exploration of the A3C Reinforcement
            # TODO: USE 1x1 kernels-bottleneck, CS231n Winter 2016: Lecture 11 from 29 minutes
            self.convolutional_layers = self._create_conv_layers()

            # This is the final fully connected layer
            self.state_embedding = slim.fully_connected(slim.flatten(self.convolutional_layers[-1]),
                                                        self.fully_connected_neuron, activation_fn=tf.nn.elu)

            self._create_dnd_variables()

            self._create_scatter_update_ops()

            self._create_gather_ops()

            self.nn_state_embeddings, self.nn_state_values = self._create_stacked_gather()

            # DND calculation
            # expand_dims() is needed to subtract the key(s) (state_embedding) from neighboring keys (Eq. 5)
            self.expand_dims = tf.expand_dims(tf.expand_dims(self.state_embedding, axis=1), axis=1)
            self.square_diff = tf.square(self.expand_dims - self.nn_state_embeddings)

            # We clip the values here, because the 0 values cause problems during backward pass (NaNs)
            self.distances = tf.sqrt(tf.clip_by_value(tf.reduce_sum(self.square_diff, axis=3),
                                                      1e-12, 1e12)) + self.delta
            self.weightings = 1.0 / self.distances
            # Normalised weightings (Eq. 2)
            self.normalised_weightings = self.weightings / tf.reduce_sum(self.weightings, axis=2, keep_dims=True)
            # (Eq. 1)
            self.squeeze = tf.squeeze(self.nn_state_values, axis=3)
            self.pred_q_values = tf.reduce_sum(self.squeeze * self.normalised_weightings, axis=2,
                                               name="predicted_Q_values")
            self.predicted_q = tf.argmax(self.pred_q_values, axis=1, name="predicted_Q_arg")

        with tf.device("/cpu:0"):
            # TODO: Check if action_index device placement is not a perf. problem (probably not)
            # This has to be an iterable, e.g.: [1, 0, 0]
            self.action_index = tf.placeholder(tf.int32, [None], name="action")
            self.action_onehot = tf.one_hot(self.action_index, self.number_of_actions, axis=-1)

        with tf.device(device):
            # Loss Function
            self.target_q = tf.placeholder(tf.float32, [None], name="target_Q")
            self.q_value = tf.reduce_sum(tf.multiply(self.pred_q_values, self.action_onehot), axis=1,
                                     name="calculated_Q_value")
            self.td_err = tf.subtract(self.target_q, self.q_value, name="td_error")
            self.total_loss = tf.square(self.td_err, name="total_loss")

            # Optimizer
            self.optimizer = tf.contrib.opt.LazyAdamOptimizer(self.adam_learning_rate).minimize(self.total_loss)

        # ----------- AUXILIARY ----------- #
        # ----------- TF related ----------- #

        # Global initialization
        with tf.device(device):
            self.init_op = tf.global_variables_initializer()
        self.session.run(self.init_op)

        # Check op for NaN checking - if needed
        # with tf.device(device):
        #     self.check_op = tf.add_check_numerics_ops()

        # Saver op
        self.saver = tf.train.Saver(max_to_keep=5)

        # ----------- Episode related containers ----------- #
        self._observation_list = []
        self._agent_input_list = []
        self._agent_input_hashes_list = []
        self._agent_action_list = []
        self._rewards_deque = deque()
        self._q_values_list = []

        # Logging and TF FileWriter
        self.log_save_directory = log_save_directory
        if self.log_save_directory:
            self.summary_writer = tf.summary.FileWriter(self.log_save_directory, graph=self.session.graph)
        self._log_hyperparameters()

        # Create discount factor vector
        self._gammas = list(map(lambda x: self.discount_factor ** x, range(self.n_step_horizon)))

        # Create epsilon decay rate (Now it is linearly decreasing between 1 and 0.001)
        self._epsilon_decay_rate = (1 - 0.001) / (self.epsilon_decay_bounds[1] - self.epsilon_decay_bounds[0])

        # Majd kibasszuk innen
        self.__options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    # This is the main function which we call in different environments during playing
    def get_action(self, processed_observation):
        # Get the agent input (frame-stacking) using the preprocessed observation
        # We also store the relevant quantities
        agent_input = self._get_agent_input(processed_observation)
        # Get the action
        action = self._get_action(agent_input)
        # Optimize if the global_step number is above optimization_start (making sure we have enough elements in the
        # replay memory and each DND)
        if self.global_step >= self.optimization_start:
            self._optimize()
            # Calculate bootstrap Q value as early as possible, so we can insert the corresponding (S, A, Q) tuple into
            # the replay memory. Because of this, the agent may sample from this example during the next _optimize()
            # call. (Intentionally)
            if len(self._rewards_deque) == self.n_step_horizon:
                q = self._calculate_bootstrapped_q_value()
                # Store (S, A, Q) in the replay memory
                self._add_to_replay_memory(q)
                # We pop the leftmost element from the rewards deque, hence the condition before
                # _calculate_bootstrapped_q_value() remains True until the episode end.
                # (Also we do not need this element anymore, since we have already used it for calculating the Q value.)
                self._rewards_deque.popleft()

        self.global_step += 1
        self.episode_step += 1

        return action

    def get_action_for_test(self, processed_observation):
        agent_input = self._get_agent_input(processed_observation)

        action = self._get_action(agent_input)
        self.episode_step += 1
        return action

    # This is the main function which we call in different environments after an episode is finished.
    def update(self):
        #  játék vége van kiszámolom a disc_rewardokat viszont az elsőnek n_hor darab rewardból
        #  a másodiknak (n_hor-1) darab rewardból, a harmadiknak (n_hor-2) darab rewardból, ésígytovább.
        #  A bootstrap value itt mindig 0 tehát a Q(N) maga a discounted reward. Majd berakosgatom a replay memoryba
        # Itt van lekezelve az, hogy a játék elején Monte-Carlo return-nel számoljuk ki a state-action value-kat.

        q_ns = self._discount(self._rewards_deque)
        self._add_to_replay_memory_episode_end(q_ns)

        # DND Lengths before modification
        dnd_lengths = self._dnd_lengths()

        actions, batch_valid_indices, batch_indices_for_ann, state_embeddings, batch_cond_vector =\
            self._tabular_like_update(self._agent_input_list, self._agent_input_hashes_list, self._agent_action_list,
                                      self._q_values_list)

        self._ann_index_update(actions, batch_valid_indices, batch_indices_for_ann, state_embeddings, batch_cond_vector,
                               dnd_lengths)
        
        self.reset_episode_related_containers()

        # Add average episode total reward to its deque
        if self.log_save_directory:
            self.windowed_average_total_reward.append(self.episode_total_reward)
            self._tensorboard_reward_writer()

        self.reset_episode_related_containers()

    def reset_episode_related_containers(self):
        self._observation_list = []
        self._agent_input_list = []
        self._agent_input_hashes_list = []
        self._agent_action_list = []
        self._rewards_deque = deque()
        self._q_values_list = []
        self.episode_step = 0
        # Increment episode number
        self.episode_number += 1
        # Set episode total reward to 0
        self.episode_total_reward = 0

    def save_action_and_reward(self, a, r):
        # Convert action to float here just for checking purposes. _check_list_ids()
        self._agent_action_list.append(float(a))
        self._rewards_deque.append(r)
        # Add step reward to episode total reward
        self.episode_total_reward += r

    def agent_save(self, path):
        self.saver.save(self.session, path + '/model_' + str(self.global_step) + '.cptk')

    def full_save(self, path):
        self.agent_save(path)
        # az LRU mappán belül hozza létre az actionokhöz tartozó .npy fájlt.
        # Ebből létre lehet hozni a "self.state_hash__tf_index" is!
        try:
            os.mkdir(path + '/LRU_' + str(self.global_step))
        except FileExistsError:
            pass
        for a, dict in self.tf_index__state_hash.items():
            np.save(path + '/LRU_' + str(self.global_step) + "/" + str(a) + '.npy', dict.items())

        self.replay_memory.save(path, self.global_step)

    def agent_load(self, path, glob_step_num):
        self.saver.restore(self.session, path + "/model_" + str(glob_step_num) + '.cptk')
        self.global_step = glob_step_num

    def full_load(self, path, glob_step_num):
        self.agent_load(path, glob_step_num)
        for a in self.action_vector:
            act_LRU = np.load(path + '/LRU_' + str(glob_step_num) + "/" + str(a) + '.npy')
            # azért reversed, hogy a lista legelső elemét rakja bele utoljára, így az lesz az MRU
            for tf_index, state_hash in reversed(act_LRU):
                self.tf_index__state_hash[a][tf_index] = state_hash
                self.state_hash__tf_index[a][state_hash] = tf_index
        # ANN index building
        dnd_keys = self.session.run(list(self.dnd_key_ops.values()))
        for act, ann in self.anns.items():
            action_index = self.action_vector.index(act)
            ann.build_index(dnd_keys[action_index][:self._dnd_length(act)])

        self.replay_memory.load(path, glob_step_num)

    # Should be a pre-processed observation
    def _get_agent_input(self, processed_observation):
        if self.episode_step == 0:
            agent_input = self._initial_frame_stacking(processed_observation)
        else:
            agent_input = self._frame_stacking(self._agent_input_list[-1], processed_observation)
        # Saving the relevant quantities
        self._observation_list.append(processed_observation)
        self._agent_input_list.append(agent_input)
        self._agent_input_hashes_list.append(hash128(agent_input))
        # self._agent_input_hashes_list.append(hash(agent_input.tobytes()))
        return agent_input

    def _optimize(self):
        # self.__run_metadata = tf.RunMetadata()

        # Get the batches from replay memory and run optimizer
        state_batch, action_batch, q_n_batch = self.replay_memory.get_batch(self.batch_size)
        action_batch_indices = [self.action_vector.index(a) for a in action_batch]
        search_keys = self.session.run(self.state_embedding,
                                       feed_dict={self.state: state_batch})
        batch_indices = self._search_ann(search_keys, 0)
        feed_dict = {self.state: state_batch, self.action_index: action_batch_indices, self.target_q: q_n_batch}
        feed_dict.update({o: k for o, k in zip(self.dnd_placeholder_ops.values(), batch_indices)})
        batch_total_loss, _ = self.session.run([self.total_loss, self.optimizer],
                                               feed_dict=feed_dict)
                                               # options=self.__options, run_metadata=self.__run_metadata)

        # self.summary_writer.add_run_metadata(self.__run_metadata, "run_data" + str(self.global_step))
        # self.summary_writer.flush()

        # Mean of the total loss for Tensorboard visualization
        if self.log_save_directory:
            self._tensorboard_loss_writer(batch_total_loss)

        # log.debug("Optimizer has been run.")

        # fetched_timeline = timeline.Timeline(self.__run_metadata.step_stats)
        # chrome_trace = fetched_timeline.generate_chrome_trace_format()
        # file = "/home/atoth/temp/lazy_adamopt_new_gather" + str(self.global_step) + ".json"
        # with open(file, "w") as f:
        #     f.write(chrome_trace)
        #     print("bugyi")

    def _get_action(self, agent_input):
        # Choose the random action
        if np.random.random_sample() < self.curr_epsilon():
            action = np.random.choice(self.action_vector)
        # Choose the greedy action
        else:
            # We expand the agent_input dimensions here to run the graph for batch_size = 1 -- action selection
            search_keys = self.session.run(self.state_embedding,
                                           feed_dict={self.state: np.expand_dims(agent_input, axis=0)})
            batch_indices = self._search_ann(search_keys, 1)
            feed_dict = {self.state: np.expand_dims(agent_input, axis=0)}
            feed_dict.update({o: k for o, k in zip(self.dnd_placeholder_ops.values(), batch_indices)})
            max_q = self.session.run(self.predicted_q, feed_dict=feed_dict)
            log.debug("Max. Q value: {}".format(max_q[0]))
            action = self.action_vector[max_q[0]]
            log.debug("Chosen action: {}".format(action))

        return action

    def _tensorboard_loss_writer(self, batch_total_loss):
        # if self.global_step == self.optimization_start:
        if self.create_list_for_total_losses:
            self.create_list_for_total_losses = False
            self._loss_list = []
            self._mean_size = 0

        self._loss_list.append(batch_total_loss)
        self._mean_size += 1
        if self._mean_size % 10 == 0:
            mean_total_loss = np.mean(self._loss_list)
            summary = tf.Summary()
            summary.value.add(tag='Total Loss', simple_value=float(mean_total_loss))
            self.summary_writer.add_summary(summary, self.global_step)
            self.summary_writer.flush()
            self._loss_list = []
            self._mean_size = 0

    def _tensorboard_reward_writer(self):
        average_windowd_episode_reward = np.mean(self.windowed_average_total_reward)
        summary = tf.Summary()
        summary.value.add(tag='Average episode reward', simple_value=float(average_windowd_episode_reward))
        self.summary_writer.add_summary(summary, self.global_step)
        self.summary_writer.flush()

    def _add_to_replay_memory(self, q, episode_end=False):
        s = self._observation_list[self.episode_step - self.n_step_horizon]
        a = self._agent_action_list[self.episode_step - self.n_step_horizon]
        # self._check_list_ids(s, a, q)
        self.replay_memory.append((s, a, q), episode_end)

    def _add_to_replay_memory_episode_end(self, q_list):
        j = len(self._rewards_deque)
        for i, (o, a, q_n) in enumerate(zip(self._observation_list[-j:], self._agent_action_list[-j:], q_list)):
            self._q_values_list.append(q_n)
            e_e = False
            if i == j - 1:
                e_e = True
            # self._check_list_ids(o, a, q_n)
            self.replay_memory.append((o, a, q_n), e_e)

    # Note that this function calculate only one Q at a time.
    def _calculate_bootstrapped_q_value(self):
        discounted_reward = np.dot(self._rewards_deque, self._gammas)
        state = [self._agent_input_list[self.episode_step]]
        search_keys = self.session.run(self.state_embedding,
                                       feed_dict={self.state: state})
        batch_indices = self._search_ann(search_keys, 0)
        feed_dict = {self.state: state}
        feed_dict.update({o: k for o, k in zip(self.dnd_placeholder_ops.values(), batch_indices)})
        bootstrap_value = np.amax(self.session.run(self.pred_q_values,
                                                   feed_dict=feed_dict))
        disc_bootstrap_value = self.discount_factor ** self.n_step_horizon * bootstrap_value
        q_value = discounted_reward + disc_bootstrap_value

        # Store calculated Q value
        self._q_values_list.append(q_value)
        return q_value

    def curr_epsilon(self):
        eps = self.initial_epsilon
        if self.epsilon_decay_bounds[0] <= self.global_step < self.epsilon_decay_bounds[1]:
            eps = self.initial_epsilon - ((self.global_step - self.epsilon_decay_bounds[0]) * self._epsilon_decay_rate)
        elif self.global_step >= self.epsilon_decay_bounds[1]:
            eps = 0.001
        return eps

    def _search_ann(self, search_keys, update_LRU_order):
        batch_indices = []
        for act, ann in self.anns.items():
            # These are the indices we get back from ANN search
            indices = ann.query(search_keys)
            # log.debug("ANN indices for action {}: {}".format(act, indices))
            # Create numpy array with full of corresponding action vector index
            # action_indices = np.full(indices.shape, self.action_vector.index(act))
            # log.debug("Action indices for action {}: {}".format(act, action_indices))
            # Riffle two arrays
            # tf_indices = self._riffle_arrays(action_indices, indices)
            batch_indices.append(indices)
            # Very important part: Modify LRU Order here
            # Doesn't work without tabular update of course!
            if update_LRU_order == 1:
                _ = [self.tf_index__state_hash[act][i] for i in indices.ravel()]
        np_batch = np.asarray(batch_indices, dtype=np.int32)
        # log.debug("Batch update indices: {}".format(np_batch))

        # Reshaping to gather_nd compatible format
        # final_indices = np.asarray([np_batch[:, j, :, :] for j in range(np_batch.shape[1])], dtype=np.int32)

        return np_batch

    def _tabular_like_update(self, states, state_hashes, actions, q_ns):
        log.debug("Tabular like update has been started.")
        # Making np arrays
        states = np.asarray(states, dtype=np.float32)
        q_ns = np.asarray(q_ns)
        actions = np.asarray(actions, dtype=np.int32)

        action_indices = np.asarray([self.action_vector.index(act) for act in actions])

        dnd_q_values = np.zeros(q_ns.shape, dtype=np.float32)
        dnd_gather_indices = np.asarray([self.state_hash__tf_index[a][sh] if sh in self.state_hash__tf_index[a]
                                         else None for sh, a in zip(state_hashes, actions)])
        # TÖRÖLNI
        for i in dnd_gather_indices:
            if i != None:
                self.seen_states_number += 1

        in_cond_vector = dnd_gather_indices != None
        # indices = np.squeeze(self._riffle_arrays(action_indices[in_cond_vector], dnd_gather_indices[in_cond_vector]),
        #                      axis=0)
        indices = self._batches_by_action(action_indices[in_cond_vector], dnd_gather_indices[in_cond_vector])

        feed_dict = {o: k for o, k in zip(self.dnd_placeholder_ops.values(), indices)}

        dnd_q_vals = self.session.run(list(self.dnd_value_gather_ops.values()), feed_dict=feed_dict)
        dnd_q_vals2 = [deque(np.squeeze(d, axis=1)) for d in dnd_q_vals]
        dnd_q_vals = [dnd_q_vals2[a].popleft() for a in action_indices[in_cond_vector]]
        dnd_q_values[in_cond_vector] = dnd_q_vals

        local_sh_dict = {a: {} for a in self.action_vector}

        # Batch means one complete game (21-points) in this context
        batch_update_values = []
        batch_indices = []
        batch_states = []
        batch_indices_for_ann = []
        batch_valid_indices = np.full(q_ns.shape, False, dtype=np.bool)
        batch_cond_vector = []
        ii = 0

        for j, (act, sh, q, state) in enumerate(zip(actions, state_hashes, q_ns, states)):
            if sh in self.state_hash__tf_index[act] and sh not in local_sh_dict[act]:
                update_value = self.tab_alpha * (q - dnd_q_values[j]) + dnd_q_values[j]
                local_sh_dict[act][sh] = (ii, update_value)

                # Add elements to lists
                batch_states.append(state)
                batch_indices.append(dnd_gather_indices[j])
                batch_update_values.append(update_value)
                batch_indices_for_ann.append(dnd_gather_indices[j])
                batch_valid_indices[j] = True
                # ANN related - Append True because it is already added to ANN points
                batch_cond_vector.append(True)
                ii += 1

            elif sh in self.state_hash__tf_index[act] and sh in local_sh_dict[act]:
                # We are not adding elements to the lists in this case
                update_value = self.tab_alpha * (q - local_sh_dict[act][sh][1]) + local_sh_dict[act][sh][1]
                ind = local_sh_dict[act][sh][0]
                batch_update_values[ind] = update_value
                local_sh_dict[act][sh] = (ind, update_value)
            else:
                if len(self.tf_index__state_hash[act]) < self.dnd_max_memory:
                    index = len(self.tf_index__state_hash[act])
                else:
                    index, old_state_hash = self.tf_index__state_hash[act].peek_last_item()
                    del self.state_hash__tf_index[act][old_state_hash]
                # LRU order stuff
                self.tf_index__state_hash[act][index] = sh
                self.state_hash__tf_index[act][sh] = index

                # Add elements to lists and update local_sh_dict
                local_sh_dict[act][sh] = (ii, q)

                batch_states.append(state)
                batch_indices.append(index)
                batch_update_values.append(q)
                batch_indices_for_ann.append(index)
                batch_valid_indices[j] = True
                batch_cond_vector.append(False)
                ii += 1

        batch_states = np.asarray(batch_states, dtype=np.float32)
        batch_indices = np.expand_dims(np.asarray(batch_indices, dtype=np.int32), axis=1)
        batch_update_values = np.asarray(batch_update_values, dtype=np.float32)
        batch_indices_for_ann = np.asarray(batch_indices_for_ann, dtype=np.int32)
        batch_cond_vector = np.asarray(batch_cond_vector, dtype=np.bool)

        # Create batch indices and update values for TensorFlow session
        # batch_indices = np.squeeze(self._riffle_arrays(action_indices[batch_valid_indices], batch_indices))
        batch_states_mod = self._batches_by_action(action_indices[batch_valid_indices], batch_states, False)
        batch_indices = self._batches_by_action(action_indices[batch_valid_indices], batch_indices, False)
        batch_update_values = self._batches_by_action(action_indices[batch_valid_indices],
                                                      np.expand_dims(batch_update_values, axis=1), False)

        # Batch tabular update
        scatter_update_key_ops = list(self.dnd_scatter_update_key_ops.values())
        scatter_update_value_ops = list(self.dnd_scatter_update_value_ops.values())
        scatter_update_ph_ops = list(self.dnd_scatter_update_placeholder_ops.values())
        scatter_update_value_ph_ops = list(self.dnd_value_update_placeholder_ops.values())
        for i, (b_s, b_i, b_u) in enumerate(zip(batch_states_mod, batch_indices, batch_update_values)):
            if len(b_s) > 0:
                ops = [scatter_update_key_ops[i], scatter_update_value_ops[i]]
                feed_dict = {self.state: b_s, scatter_update_ph_ops[i]: b_i, scatter_update_value_ph_ops[i]: b_u}
                self.session.run(ops, feed_dict=feed_dict)

        state_embeddings = self.session.run(self.state_embedding, feed_dict={self.state: batch_states})

        log.debug("Tabular like update has been run.")

        return actions, batch_valid_indices, batch_indices_for_ann, state_embeddings, batch_cond_vector

    def _ann_index_update(self, actions, batch_valid_indices, batch_indices_for_ann, state_embeddings,
                          batch_cond_vector, dnd_lengths):
        index_rebuild = not bool(self.episode_number % self.ann_rebuild_freq)
        # FLANN Add point - every batch
        if not index_rebuild:
            for a in self.action_vector:
                act_cond = actions[batch_valid_indices] == a
                self.anns[a].update_ann(batch_indices_for_ann[act_cond], state_embeddings[act_cond],
                                        batch_cond_vector[act_cond], dnd_lengths[self.action_vector.index(a)])

        # FLANN index rebuild, if index_rebuild = True
        if index_rebuild:
            dnd_keys = self.session.run(list(self.dnd_key_ops.values()))
            for act, ann in self.anns.items():
                action_index = self.action_vector.index(act)
                # Ez a jó (kövi sor)
                ann.build_index(dnd_keys[action_index][:self._dnd_length(act)])

    def _save_q_value(self, q):
        self._q_values_list.append(q)

    def _discount(self, x):
        a = np.asarray(x)
        return lfilter([1], [1, -self.discount_factor], a[::-1], axis=0)[::-1]

    def _dnd_lengths(self):
        return [len(self.tf_index__state_hash[a]) for a in self.action_vector]

    def _dnd_length(self, a):
        return len(self.tf_index__state_hash[a])

    def _create_conv_layers(self):
        """
        Create convolutional layers in the Tensorflow graph according to the hyperparameters, using Tensorflow slim
        library.

        Returns
        -------
        conv_layers: list
            The list of convolutional operations.

        """
        lengths_set = {len(o) for o in (self._num_outputs, self._kernel_size, self._stride)}
        if len(lengths_set) != 1:
            msg = "The lengths of the conv. layers params vector should be same. Lengths: {}, Vectors: {}".format(
                [len(o) for o in (self._num_outputs, self._kernel_size, self._stride)],
                (self._num_outputs, self._kernel_size, self._stride))
            raise ValueError(msg)
        conv_layers = []
        inputs = [self.state]
        for i, (num_out, kernel, stride) in enumerate(zip(self._num_outputs, self._kernel_size, self._stride)):
            layer = slim.conv2d(activation_fn=tf.nn.elu, inputs=inputs[i], num_outputs=num_out,
                                kernel_size=kernel, stride=stride, padding='SAME')
            conv_layers.append(layer)
            inputs.append(layer)
        return conv_layers

    def _create_dnd_variables(self):
        with tf.variable_scope("dnd_keys"):
            for a in self.action_vector:
                k = tf.get_variable("dnd_keys_for_action_" + str(a), (self.dnd_max_memory, self.fully_connected_neuron),
                                    dtype=tf.float32, initializer=tf.zeros_initializer)
                self.dnd_key_ops[a] = k

        with tf.variable_scope("dnd_values"):
            for a in self.action_vector:
                v = tf.get_variable("dnd_values_for_action_" + str(a), (self.dnd_max_memory, 1),
                                    dtype=tf.float32, initializer=tf.zeros_initializer)
                self.dnd_value_ops[a] = v

    def _create_gather_ops(self):
        with tf.variable_scope("dnd_gather_ops"):
            for a, k in self.dnd_key_ops.items():
                self.dnd_placeholder_ops[a] = tf.placeholder(tf.int32, None, name="gather_ph_for_action_" + str(a))
                self.dnd_key_gather_ops[a] = tf.gather(k, self.dnd_placeholder_ops[a], axis=0,
                                                       name="key_gather_op_for_action_" + str(a))
            for a, v in self.dnd_value_ops.items():
                self.dnd_value_gather_ops[a] = tf.gather(v, self.dnd_placeholder_ops[a], axis=0,
                                                         name="val_gather_op_for_action_" + str(a))

    def _create_stacked_gather(self):
        key_gather_ops = [op for op in self.dnd_key_gather_ops.values()]
        value_gather_ops = [op for op in self.dnd_value_gather_ops.values()]
        nn_state_embeddings = tf.stack(key_gather_ops, axis=1, name="nn_state_embeddings")
        nn_state_values = tf.stack(value_gather_ops, axis=1, name="nn_state_values")
        return nn_state_embeddings, nn_state_values

    def _create_scatter_update_ops(self):
        with tf.variable_scope("dnd_scatter_update"):
            for a in self.action_vector:
                self.dnd_scatter_update_placeholder_ops[a] = tf.placeholder(tf.int32, None,
                                                                            name="update_ind_ph_for_action_" + str(a))
                self.dnd_value_update_placeholder_ops[a] = tf.placeholder(tf.float32, None,
                                                                          name="val_update_ph_for_action_" + str(a))
                self.dnd_scatter_update_key_ops[a] = tf.scatter_nd_update(self.dnd_key_ops[a],
                                                                          self.dnd_scatter_update_placeholder_ops[a],
                                                                          self.state_embedding,
                                                                          name="key_update_op_for_action_" + str(a))
                self.dnd_scatter_update_value_ops[a] = tf.scatter_nd_update(self.dnd_value_ops[a],
                                                                            self.dnd_scatter_update_placeholder_ops[a],
                                                                            self.dnd_value_update_placeholder_ops[a],
                                                                            name="val_update_op_for_action_" + str(a))

    @staticmethod
    def _riffle_arrays(array_1, array_2):
        if len(array_1.shape) == 1:
            array_1 = np.expand_dims(array_1, axis=0)
            array_2 = np.expand_dims(array_2, axis=0)

        tf_indices = np.empty([array_1.shape[0], array_1.shape[1] * 2], dtype=array_1.dtype)
        # Riffle the action indices with ann output indices
        tf_indices[:, 0::2] = array_1
        tf_indices[:, 1::2] = array_2
        return tf_indices.reshape((array_1.shape[0], array_1.shape[1], 2))

    def _batches_by_action(self, array_1, array_2, use_deque=True):
        indices = [deque() for _ in self.action_vector]
        for a, i in zip(array_1, array_2):
            indices[a].append(i)
        if use_deque:
            return indices
        else:
            return [np.asarray(array) for array in indices]

    def _initial_frame_stacking(self, processed_obs):
        return np.stack((processed_obs, ) * self.frame_stacking_number, axis=2)

    @staticmethod
    def _frame_stacking(s_t, o_t):  # Ahol az "s_t" a korábban stackkelt 4 frame, "o_t" pedig az új observation
        s_t1 = np.append(s_t[:, :, 1:], np.expand_dims(o_t, axis=2), axis=2)
        return s_t1

    def _log_hyperparameters(self):
        log.info("The hyperparameters of the agent are:\n"
                 "Optimizer parameters\n"
                 "--------------------\n"
                 "Learning rate: {lr}\n"
                 "Batch size for optimization: {bs}\n"
                 "Global step number when optimization starts: {os}\n"
                 "\n"
                 "Tabular parameters\n"
                 "------------------\n"
                 "Q update learning rate: {qlr}\n"
                 "DND maximum memory: {dnd}\n"
                 "\n"
                 "Reinforcement learning parameters\n"
                 "---------------------------------\n"
                 "N-step horizon: {n}\n"
                 "Discount factor: {df}\n"
                 "Starting epsilon: {init_e}\n"
                 "Final epsilon: 0.001\n"
                 "Epsilon is linearly decaying between global step number {eps_d[0]} and {eps_d[1]}\n"
                 "\n"
                 "Convolutional layer parameters\n"
                 "------------------------------\n"
                 "Input shape: {inp_s}\n"
                 "Fully connected neuron number: {fcn}\n"
                 "Kernel sizes: {ks}\n"
                 "Strides: {ss}\n"
                 "Number of outputs of each layer: {num_o}\n"
                 "\n"
                 "Environment specific parameters\n"
                 "-------------------------------\n"
                 "Available actions: {act}\n"
                 "Frame stacking number: {fs}\n"
                 "\n"
                 "Approx. Nearest Neighbor search parameters\n"
                 "------------------------------------------\n"
                 "ANN update frequency(episode number): {auf}\n"
                 "Nearest Neighbor search number: {nn}".format(lr=self.adam_learning_rate, bs=self.batch_size,
                                                               os=self.optimization_start, qlr=self.tab_alpha,
                                                               dnd=self.dnd_max_memory, n=self.n_step_horizon,
                                                               df=self.discount_factor, init_e=self.initial_epsilon,
                                                               eps_d=self.epsilon_decay_bounds, inp_s=self._input_shape,
                                                               fcn=self.fully_connected_neuron, ks=self._kernel_size,
                                                               ss=self._stride, num_o=self._num_outputs,
                                                               act=self.action_vector, fs=self.frame_stacking_number,
                                                               auf=self.ann_rebuild_freq, nn=self.neighbor_number))

    @staticmethod
    def _create_tf_session():
        # return tf.Session(config=tf.ConfigProto(log_device_placement=True))
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        return sess

    def _check_list_ids(self, s, a, q):
        def get_index(l, o):
            for i, j in enumerate(l):
                if id(j) == id(o):
                    return i
        i_1 = get_index(self._observation_list, s)
        i_2 = get_index(self._agent_action_list, a)
        i_3 = get_index(self._q_values_list, q)
        if not (i_1 == i_2 and i_2 == i_3):
            raise ValueError("The indices are wrong: {}".format((i_1, i_2, i_3)))


class AnnSearch:

    def __init__(self, neighbors_number, dnd_max_memory, action):
        self.ann = FLANN()
        self.neighbors_number = neighbors_number
        self._ann_index__tf_index = {}
        self._ann_index__tf_index_v2 = {}
        self.dnd_max_memory = int(dnd_max_memory)
        self._removed_points = 0
        self.flann_params = None
        # For logging purposes
        self.action = action

    def add_state_embedding(self, state_embedding):
        self.ann.add_points(state_embedding)

    def update_ann(self, tf_var_dnd_indices, state_embeddings, cond_vector, dnd_actual_length):
        # A tf_var_dnd_index alapján kell törölnünk a Flann indexéből. Ez csak abban az esetben fog
        # kelleni, ha nincs index build és egy olyan index jön be, amihez tartozó state_embeddeinget már egyszer hozzáadtam.

        # Ha láttuk már a pontot akkor ki kell törölni, mert a state hash-hehz tartozó state embedding érték megváltozott
        # és azt tároljuk ANN-ben
        flann_indices_seen = []
        for tf_var_dnd_index in tf_var_dnd_indices[cond_vector]:
            if tf_var_dnd_index in self._ann_index__tf_index.values():
                index = [k for k, v in self._ann_index__tf_index.items() if v == tf_var_dnd_index][0]
            else:
                index = tf_var_dnd_index
            flann_indices_seen.append(index)
        # flann_indices_seen = [k for k, v in self._ann_index__tf_index.items() if v in tf_var_dnd_indices[cond_vector]]
        self.ann.remove_points(flann_indices_seen)

        for i, tf_var_dnd_index in enumerate(tf_var_dnd_indices[cond_vector]):
            self._ann_index__tf_index[dnd_actual_length + self._removed_points + i] = tf_var_dnd_index

        # Itt adjuk hozzá a FLANN indexéhez a már látott state hash-hez
        if len(state_embeddings[cond_vector]) != 0:
            self.add_state_embedding(state_embeddings[cond_vector])

        self._removed_points += len(flann_indices_seen)

        # Ha nem láttuk és tele vagyunk
        # debug2_list = []
        counter = 0
        #print(len(tf_var_dnd_indices[~cond_vector]))
        for i, tf_var_dnd_index in enumerate(tf_var_dnd_indices[~cond_vector]):
            if dnd_actual_length + i >= self.dnd_max_memory:
                if tf_var_dnd_index in self._ann_index__tf_index.values():
                    index = [k for k, v in self._ann_index__tf_index.items() if v == tf_var_dnd_index][0]
                else:
                    index = tf_var_dnd_index

                # ez a rész itt még zsivány, nem fölfele
                self.ann.remove_point(index)
                self._ann_index__tf_index[dnd_actual_length + self._removed_points + counter] = tf_var_dnd_index

                self._removed_points += 1

            else:
                self._ann_index__tf_index[dnd_actual_length + self._removed_points + counter] = tf_var_dnd_index

                counter += 1

        self.add_state_embedding(state_embeddings[~cond_vector])
        self._ann_index__tf_index_v2.update(self._ann_index__tf_index)

    def build_index(self, tf_variable_dnd):
        self.flann_params = self.ann.build_index(tf_variable_dnd, algorithm="kdtree", target_precision=1)
        self._ann_index__tf_index = {}
        self._removed_points = 0
        # log.info("ANN index has been rebuilt for action {}.".format(self.action))
        self._ann_index__tf_index_v2 = {i: i for i in range(len(tf_variable_dnd))}

    def query(self, state_embeddings):
        indices, _ = self.ann.nn_index(state_embeddings, num_neighbors=self.neighbors_number,
                                       checks=self.flann_params["checks"])
        # tf_var_dnd_indices = [[self._ann_index__tf_index[j] if j in self._ann_index__tf_index else j for j in index_row]
        #                       for index_row in indices]
        int64_indices = np.asarray(indices, dtype=np.int64)
        tf_var_dnd_indices = [[self._ann_index__tf_index_v2[j] for j in index_row] for index_row in int64_indices]

        return np.asarray(tf_var_dnd_indices, dtype=np.int32)


def setup_logging(level=logging.INFO, is_stream_handler=True, is_file_handler=False, file_handler_filename=None):
    log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if is_stream_handler:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        log.addHandler(ch)

    if file_handler_filename:
        fh = logging.FileHandler(file_handler_filename)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        log.addHandler(fh)
