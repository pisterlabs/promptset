import sys, os, csv, gym
import time
import multiprocessing
from collections import deque
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from utils.replay_buffer import MultiWorkerReplayBuffer, PrioritizedMultiWorkerReplayBuffer
from utils.replay_buffer import DoublePrioritizedReplayWrapper
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.policies import SACPolicy
from stable_baselines import logger
from utils.eval_stack import eval_model


def get_vars(scope):
    """
    Alias for get_trainable_vars
    :param scope: (str)
    :return: [tf Variable]
    """
    return tf_util.get_trainable_vars(scope)


class SAC_SIR(OffPolicyRLModel):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup) and from the Softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
    :param policy: (SACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param ent_coef: (str or float) Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_update_interval: (int) update the target network every `target_network_update_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param target_entropy: (str or float) target entropy when learning ent_coef (ent_coef = 'auto')
    :param action_noise: (ActionNoise) the action noise type (None by default), this can help
        for hard exploration problem. Cf DDPG for the different action noise type.
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for SAC normally but can help exploring when using HER + SAC.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on SAC logging for now
    """

    def __init__(self, policy, env, trained_sac_model=None, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 priority_buffer=False, alpha=0.6,
                 learning_starts=100, train_freq=1, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=1,
                 gradient_steps=1, target_entropy='auto', action_noise=None,
                 random_exploration=0.0, n_subgoal=2, start_augment_time=0,
                 aug_env=None, eval_env=None, curriculum=False, imitation_coef=5,
                 sequential=False,
                 verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False):

        super(SAC_SIR, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                      policy_base=SACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs)

        self.aug_env = aug_env
        self.eval_env = eval_env
        self.trained_sac_model = trained_sac_model
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        # In the original paper, same learning rate is used for all networks
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration
        self.n_subgoal = n_subgoal
        self.start_augment_time = start_augment_time
        self.priority_buffer = priority_buffer
        self.alpha = alpha
        self.curriculum = curriculum
        self.imitation_coef = imitation_coef
        self.sequential = sequential

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.augment_replay_buffer = None
        self.combined_replay_buffer = None
        self.episode_reward = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None
        self.num_aug_steps = 0

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = self.deterministic_action * np.abs(self.action_space.low)
        return policy.obs_ph, self.actions_ph, deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                n_cpu = multiprocessing.cpu_count()
                if sys.platform == 'darwin':
                    n_cpu //= 2
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                if hasattr(self.env, "env") and isinstance(self.env.env, VecEnv):
                    self.n_envs = self.env.env.num_envs
                else:
                    self.n_envs = 1

                if self.priority_buffer:
                    self.replay_buffer = PrioritizedMultiWorkerReplayBuffer(self.buffer_size, self.alpha,
                                                                            num_workers=self.env.env.num_envs, gamma=self.gamma)
                    self.augment_replay_buffer = PrioritizedMultiWorkerReplayBuffer(self.buffer_size, self.alpha,
                                                                                    num_workers=1, gamma=self.gamma)
                else:
                    self.replay_buffer = MultiWorkerReplayBuffer(self.buffer_size, num_workers=self.n_envs, gamma=self.gamma)
                    self.augment_replay_buffer = MultiWorkerReplayBuffer(self.buffer_size, num_workers=1,
                                                                         gamma=self.gamma)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
                                                     **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy.obs_ph
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    self.action_target = self.target_policy.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.importance_weight_ph = tf.placeholder(tf.float32, shape=(None,), name="weights")
                    # self.sum_rs_ph = tf.placeholder(tf.float32, shape=(None, 1), name="sum_rs")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probabilty of actions taken by the policy
                    self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Monitor the entropy of the policy,
                    # this is not used for training
                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                    #  Use two Q-functions to improve performance by reducing overestimation bias.
                    qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                     create_qf=True, create_vf=True)
                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                    policy_out, create_qf=True, create_vf=False,
                                                                    reuse=True)

                    # Target entropy is used when learning the entropy coefficient
                    if self.target_entropy == 'auto':
                        # automatically set target entropy if needed
                        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
                    else:
                        # Force conversion
                        # this will also throw an error for unexpected string
                        self.target_entropy = float(self.target_entropy)

                    # The entropy coefficient or entropy can be learned automatically
                    # see Automating Entropy Adjustment for Maximum Entropy RL section
                    # of https://arxiv.org/abs/1812.05905
                    if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                        # Default initial value of ent_coef when learned
                        init_value = 1.0
                        if '_' in self.ent_coef:
                            init_value = float(self.ent_coef.split('_')[1])
                            assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                        self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                            initializer=np.log(init_value).astype(np.float32))
                        self.ent_coef = tf.exp(self.log_ent_coef)
                    else:
                        # Force conversion to float
                        # this will throw an error if a malformed string (different from 'auto')
                        # is passed
                        self.ent_coef = float(self.ent_coef)

                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                         create_qf=False, create_vf=True)
                    self.value_target = value_target

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                    # Target for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * self.value_target
                    )

                    # Compute Q-Function loss
                    # TODO: test with huber loss (it would avoid too high values)
                    qf1_loss = 0.5 * tf.reduce_mean(self.importance_weight_ph * (q_backup - qf1) ** 2)
                    qf2_loss = 0.5 * tf.reduce_mean(self.importance_weight_ph * (q_backup - qf2) ** 2)

                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.ent_coef, float):
                        ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)
                    # self.is_demo_ph = tf.placeholder(tf.float32, shape=(None,), name='is_demo')
                    # Behavior cloning loss
                    # policy_imitation_loss = tf.reduce_mean(
                    #     self.is_demo_ph * tf.reduce_mean(tf.square(self.deterministic_action - self.actions_ph),
                    #                                      axis=-1) * tf.stop_gradient(tf.cast(tf.greater(qf1, qf1_pi), tf.float32)))
                    # Self imitation style loss
                    self.logpac_op = logp_ac = self.logpac(self.actions_ph)
                    policy_imitation_loss = tf.reduce_mean(
                        (-logp_ac * tf.stop_gradient(tf.nn.relu(qf1 - value_fn))))

                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the gaussian parameters
                    # this is not used for now
                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    policy_loss = policy_kl_loss + self.imitation_coef * policy_imitation_loss


                    # Target for value fn regression
                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                    value_loss = 0.5 * tf.reduce_mean(self.importance_weight_ph * (value_fn - v_backup) ** 2)

                    values_losses = qf1_loss + qf2_loss + value_loss

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))

                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = get_vars('model/values_fn')

                    source_params = get_vars("model/values_fn/vf")
                    target_params = get_vars("target/values_fn/vf")

                    # Polyak averaging for target variables
                    self.target_update_op = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies([policy_train_op]):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy', 'demo_ratio']
                        # All ops to call during one training step
                        self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         value_loss, qf1, qf2, value_fn, logp_pi,
                                         self.entropy, policy_train_op, train_values_op]

                        # Add entropy coefficient optimization operation if needed
                        if ent_coef_loss is not None:
                            with tf.control_dependencies([train_values_op]):
                                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                self.infos_names += ['ent_coef_loss', 'ent_coef']
                                self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('value_loss', value_loss)
                    tf.summary.scalar('entropy', self.entropy)
                    if ent_coef_loss is not None:
                        tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                        tf.summary.scalar('ent_coef', self.ent_coef)

                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = get_vars("model")
                self.target_params = get_vars("target/values_fn/vf")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate):
        def safe_concat(arr1, arr2):
            if len(arr1) == 0:
                return arr2
            if len(arr2) == 0:
                return arr1
            return np.concatenate([arr1, arr2])

        if self.priority_buffer:
            batch1, batch2 = self.combined_replay_buffer.sample(self.batch_size, beta=0.4)
            batch1_obs, batch1_actions, batch1_rewards, batch1_next_obs, batch1_dones, batch1_weights, batch1_idxes = batch1
            batch2_obs, batch2_actions, batch2_rewards, batch2_next_obs, batch2_dones, batch2_weights, batch2_idxes = batch2
            batch_obs = safe_concat(batch1_obs, batch2_obs)
            batch_actions = safe_concat(batch1_actions, batch2_actions)
            batch_rewards = safe_concat(batch1_rewards, batch2_rewards)
            batch_next_obs = safe_concat(batch1_next_obs, batch2_next_obs)
            batch_dones = safe_concat(batch1_dones, batch2_dones)
            # batch_sumrs = safe_concat(batch1_sumrs, batch2_sumrs)
            batch_weights = safe_concat(batch1_weights, batch2_weights)
            batch_is_demo = safe_concat(np.zeros(batch1_obs.shape[0], dtype=np.float32),
                                        np.ones(batch2_obs.shape[0], dtype=np.float32))
            demo_ratio = batch2_obs.shape[0] / batch_obs.shape[0]
        else:
            # Uniform sampling
            n_augment = int(self.batch_size * (len(self.augment_replay_buffer) /
                                               (len(self.augment_replay_buffer) + len(self.replay_buffer))))
            if n_augment > 0 and self.augment_replay_buffer.can_sample(n_augment):
                batch1 = self.replay_buffer.sample(self.batch_size - n_augment)
                batch2 = self.augment_replay_buffer.sample(n_augment)
                batch1_obs, batch1_actions, batch1_rewards, batch1_next_obs, batch1_dones = batch1
                batch2_obs, batch2_actions, batch2_rewards, batch2_next_obs, batch2_dones = batch2
                batch_obs = safe_concat(batch1_obs, batch2_obs)
                batch_actions = safe_concat(batch1_actions, batch2_actions)
                batch_rewards = safe_concat(batch1_rewards, batch2_rewards)
                batch_next_obs = safe_concat(batch1_next_obs, batch2_next_obs)
                batch_dones = safe_concat(batch1_dones, batch2_dones)
                # batch_sumrs = safe_concat(batch1_sumrs, batch2_sumrs)
                batch_is_demo = safe_concat(np.zeros(batch1_obs.shape[0], dtype=np.float32),
                                            np.ones(batch2_obs.shape[0], dtype=np.float32))
                demo_ratio = batch2_obs.shape[0] / batch_obs.shape[0]
            else:
                batch = self.replay_buffer.sample(self.batch_size)
                batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch
                batch_is_demo = np.zeros(batch_obs.shape[0], dtype=np.float32)
                demo_ratio = 0.0
            batch_weights = np.ones(batch_obs.shape[0])
        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate,
            self.importance_weight_ph: batch_weights,
        }
        if hasattr(self, 'is_demo_ph'):
            feed_dict[self.is_demo_ph] = batch_is_demo

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + self.step_ops + [self.value_target], feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.step_ops + [self.value_target], feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        entropy = values[4]

        # update priority here
        if self.priority_buffer:
            qf1 = values[0]
            value_target = values[-1]
            batch_rewards = np.reshape(batch_rewards, (self.batch_size, -1))
            batch_dones = np.reshape(batch_dones, (self.batch_size, -1))
            priorities = batch_rewards + (1 - batch_dones) * self.gamma * value_target - qf1
            priorities = np.abs(priorities) + 1e-4
            priorities = np.squeeze(priorities, axis=-1).tolist()
            if len(batch1_idxes):
                self.replay_buffer.update_priorities(batch1_idxes, priorities[:len(batch1_idxes)])
            if len(batch2_idxes):
                self.augment_replay_buffer.update_priorities(batch2_idxes, priorities[-len(batch2_idxes):])

        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-3:-1]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, demo_ratio, ent_coef_loss, ent_coef

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, demo_ratio

    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)
            self.augment_replay_buffer = replay_wrapper(self.augment_replay_buffer)
            if self.priority_buffer:
                self.replay_buffer.set_model(self)
                self.augment_replay_buffer.set_model(self)
                self.combined_replay_buffer = DoublePrioritizedReplayWrapper(self.replay_buffer.replay_buffer, self.augment_replay_buffer.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn(seed)
            self.env_id = self.env.env.get_attr('spec')[0].id
            self.goal_dim = self.aug_env.get_attr('goal')[0].shape[0]
            self.obs_dim = self.aug_env.observation_space.shape[0] - 2 * self.goal_dim
            self.noise_mag = self.aug_env.get_attr('size_obstacle')[0][1]
            self.n_object = self.aug_env.get_attr('n_object')[0]
            self.reward_type = self.aug_env.get_attr('reward_type')[0]

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            select_subgoal_time = 0.

            start_time = time.time()
            episode_rewards = [[0.0] for _ in range(self.env.env.num_envs)]
            episode_successes = [[] for _ in range(self.env.env.num_envs)]
            if self.action_noise is not None:
                self.action_noise.reset()
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            pp_sr_buf = deque(maxlen=5)
            stack_sr_buf = deque(maxlen=5)
            n_updates = 0
            start_decay = total_timesteps
            # TODO: should set task_array before reset
            if self.sequential and 'FetchStack' in self.env_id:
                current_max_nobject = 2
                self.env.env.env_method('set_task_array', [[(2, 0), (2, 1), (1, 0)]] * self.env.env.num_envs)
                print('Set task_array to ', self.env.env.get_attr('task_array')[0])
                self.env.env.env_method('set_random_ratio', [0.7] * self.env.env.num_envs)
            if 'FetchStack' in self.env_id and self.curriculum:
                self.start_augment_time = np.inf
            obs = self.env.reset()
            infos_values = []

            self.ep_state_buf = [[] for _ in range(self.n_envs)]
            self.ep_transition_buf = [[] for _ in range(self.n_envs)]
            if 'FetchStack' in self.env_id:
                self.ep_current_nobject = [[] for _ in range(self.n_envs)]
                self.ep_selected_objects = [[] for _ in range(self.n_envs)]
                self.ep_task_mode = [[] for _ in range(self.n_envs)]
                self.ep_tower_height = [[] for _ in range(self.n_envs)]
            self.restart_steps = []  # Every element should be scalar
            self.subgoals = []  # Every element should be [*subgoals, ultimate goal]
            self.restart_states = []  # list of (n_candidate) states
            self.transition_storage = []  # every element is list of tuples. length of every element should match restart steps
            if 'FetchStack' in self.env_id:
                self.current_nobject = []
                self.selected_objects = []
                self.task_mode = []

            # For filtering subgoals
            self.mean_value_buf = deque(maxlen=500)

            num_augment_ep_buf = deque(maxlen=100)
            num_success_augment_ep_buf = deque(maxlen=100)

            def convert_dict_to_obs(dict_obs):
                assert isinstance(dict_obs, dict)
                return np.concatenate([dict_obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])

            def augment_cond():
                if 'FetchStack' in self.env_id:
                    if (not infos[idx]['is_success']) and task_modes[idx] == 1 and current_nobjects[idx] >= 2:
                        return True
                    return False
                elif 'MasspointMaze' in self.env_id or 'MasspointPushDoubleObstacle' in self.env_id :
                    if not infos[idx]['is_success']:
                        return True
                    return False
                else:
                    if (not infos[idx]['is_success']) and np.argmax(goal[3:]) == 0:
                        return True
                    return False

            def log_debug_value(value1, value2, goal_idx, is_success):
                if not os.path.exists(os.path.join(logger.get_dir(), 'debug_value.csv')):
                    with open(os.path.join(logger.get_dir(), 'debug_value.csv'), 'a', newline='') as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
                        title = ['value1', 'value2', 'goal_idx', 'is_success', 'num_timesteps']
                        csvwriter.writerow(title)
                with open(os.path.join(logger.get_dir(), 'debug_value.csv'), 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
                    data = [value1, value2, goal_idx, int(is_success), self.num_timesteps]
                    csvwriter.writerow(data)


            for step in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                if self.curriculum and step % 3000 == 0:
                    if 'FetchStack' in self.env.env.get_attr('spec')[0].id:
                        # Stacking
                        pp_sr = eval_model(self.eval_env, self, current_max_nobject if self.sequential else self.n_object, 1.0,
                                           init_on_table=(self.env.env.get_attr('spec')[0].id=='FetchStack-v2'))
                        pp_sr_buf.append(pp_sr)
                        stack_sr = eval_model(self.eval_env, self, current_max_nobject if self.sequential else self.n_object, 0.0,
                                              init_on_table=(self.env.env.get_attr('spec')[0].id=='FetchStack-v2'))
                        stack_sr_buf.append(stack_sr)
                        print('Pick-and-place success rate', np.mean(pp_sr_buf))
                        if self.sequential:
                            if self.env.env.get_attr('random_ratio')[0] > 0.5 and np.mean(pp_sr_buf) > 0.8:
                                _ratio = 0.3
                                # Force start augment after mastering pick-and-place on 2 objs
                                if current_max_nobject == 2:
                                    self.start_augment_time = self.num_timesteps
                            elif self.env.env.get_attr('random_ratio')[0] < 0.5 and current_max_nobject < self.n_object \
                                    and np.mean(stack_sr_buf) > 1 / current_max_nobject:
                                _ratio = 0.7
                                current_max_nobject += 1
                                previous_task_array = self.env.env.get_attr('task_array')[0]
                                self.env.env.env_method('set_task_array', [
                                    previous_task_array + [(current_max_nobject, j) for j in
                                                           range(current_max_nobject)]] * self.env.env.num_envs)

                                print('Set task_array to', self.env.env.get_attr('task_array')[0])
                            else:
                                _ratio = self.env.env.get_attr('random_ratio')[0]
                        else:
                            if start_decay == total_timesteps and np.mean(pp_sr_buf) > 0.8:
                                start_decay = step
                                # Force start augment after mastering pick-and-place on env.n_object objs
                                self.start_augment_time = self.num_timesteps
                            _ratio = np.clip(0.7 - (step - start_decay) / 2e6, 0.3, 0.7)  # from 0.7 to 0.3
                    elif 'FetchPushWallObstacle' in self.env_id:
                        _ratio = max(1.0 - step / total_timesteps, 0.0)
                    else:
                        raise NotImplementedError
                    self.env.env.env_method('set_random_ratio', [_ratio] * self.env.env.num_envs)
                    print('Set random_ratio to', self.env.env.get_attr('random_ratio')[0])

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if (self.num_timesteps < self.learning_starts
                    or np.random.rand() < self.random_exploration):
                    # No need to rescale when sampling random action
                    rescaled_action = np.stack([self.env.action_space.sample() for _ in range(self.env.env.num_envs)],
                                               axis=0)
                    action = rescaled_action
                else:
                    action = self.policy_tf.step(obs, deterministic=False)
                    # Add noise to the action (improve exploration,
                    # not needed in general)
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert action.shape == (self.env.env.num_envs, ) + self.env.action_space.shape

                internal_states = self.env.env.env_method('get_state')
                if 'FetchStack' in self.env_id:
                    current_nobjects = self.env.env.get_attr('current_nobject')
                    selected_objects = self.env.env.get_attr('selected_objects')
                    task_modes = self.env.env.get_attr('task_mode')
                    tower_height = self.env.env.get_attr('tower_height')
                for i in range(self.n_envs):
                    self.ep_state_buf[i].append(internal_states[i])
                    if 'FetchStack' in self.env_id:
                        self.ep_current_nobject[i].append(current_nobjects[i])
                        self.ep_selected_objects[i].append(selected_objects[i])
                        self.ep_task_mode[i].append(task_modes[i])
                        self.ep_tower_height[i].append(tower_height[i])
                    
                new_obs, rewards, dones, infos = self.env.step(rescaled_action)
                next_obs = new_obs.copy()
                for i in range(self.n_envs):
                    if dones[i]:
                        next_obs[i] = self.env.convert_dict_to_obs(infos[i]['terminal_observation'])
                    self.ep_transition_buf[i].append((obs[i], action[i], rewards[i], next_obs[i], dones[i]))
                
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, rewards, next_obs, dones)
                obs = new_obs
                
                for idx, _done in enumerate(dones):
                    episode_rewards[idx][-1] += rewards[idx]
                    if _done:
                        episode_rewards[idx].append(0.0)
                        maybe_is_success = infos[idx].get('is_success')
                        if maybe_is_success is not None:
                            episode_successes[idx].append(float(maybe_is_success))

                # Retrieve reward and episode length if using Monitor wrapper
                for _info in infos:
                    maybe_ep_info = _info.get('episode')
                    if maybe_ep_info is not None:
                        ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.reshape(rewards, (self.env.env.num_envs, -1))
                    ep_done = np.reshape(dones, (self.env.env.num_envs, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, self.num_timesteps)


                # episode augmentation
                for idx, done in enumerate(dones):
                    if self.num_timesteps >= self.start_augment_time and done:
                        goal = self.ep_transition_buf[idx][0][0][-self.goal_dim:]
                        if augment_cond():
                            # Do augmentation
                            # Sample start step and perturbation
                            select_subgoal_time0 = time.time()
                            _restart_steps, _subgoals = self.select_subgoal(self.ep_transition_buf[idx],
                                                                            k=self.n_subgoal,
                                                                            tower_height=self.ep_tower_height[idx] if 'FetchStack' in self.env_id else None)
                            select_subgoal_time += (time.time() - select_subgoal_time0)
                            # print('select subgoal time', select_subgoal_time)
                            assert isinstance(_restart_steps, np.ndarray)
                            assert isinstance(_subgoals, np.ndarray)
                            for j in range(_restart_steps.shape[0]):
                                self.restart_steps.append(_restart_steps[j])
                                self.subgoals.append([np.array(_subgoals[j]), goal.copy()])
                                assert len(self.subgoals[-1]) == 2
                                self.restart_states.append(self.ep_state_buf[idx][_restart_steps[j]])
                                self.transition_storage.append(self.ep_transition_buf[idx][:_restart_steps[j]])
                                if 'FetchStack' in self.env_id:
                                    self.current_nobject.append(self.ep_current_nobject[idx][0])
                                    self.selected_objects.append(self.ep_selected_objects[idx][0])
                                    self.task_mode.append(self.ep_task_mode[idx][0])
                    if done:
                        self.ep_state_buf[idx] = []
                        self.ep_transition_buf[idx] = []
                        if 'FetchStack' in self.env_id:
                            self.ep_current_nobject[idx] = []
                            self.ep_selected_objects[idx] = []
                            self.ep_task_mode[idx] = []
                            self.ep_tower_height[idx] = []

                def switch_subgoal(switch_goal_flag, current_obs):
                    for idx, flag in enumerate(switch_goal_flag):
                        if flag:
                            env_subgoals[idx].pop(0)
                            # self.aug_env.set_attr('goal', env_subgoals[idx][0], indices=idx)
                            self.aug_env.env_method('set_goal', [env_subgoals[idx][0]], indices=idx)
                            if 'FetchStack' in self.env_id:
                                # Use stack as ultimate task.
                                assert self.task_mode[idx] == 1
                                self.aug_env.env_method('set_task_mode', [self.task_mode[idx]], indices=idx)
                            switch_goal_flag[idx] = False
                            current_obs[idx] = convert_dict_to_obs(self.aug_env.env_method('get_obs', indices=idx)[0])

                while (len(self.restart_steps) >= self.aug_env.num_envs):
                    # TODO Hard work here
                    env_restart_steps = self.restart_steps[:self.aug_env.num_envs]
                    env_subgoals = self.subgoals[:self.aug_env.num_envs]
                    temp_subgoals = [goals[-2] for goals in env_subgoals].copy()
                    ultimate_goals = [goals[-1] for goals in env_subgoals]
                    env_storage = self.transition_storage[:self.aug_env.num_envs]
                    env_increment_storage = [[] for _ in range(self.aug_env.num_envs)]
                    self.aug_env.env_method('set_goal', [env_subgoals[idx][0] for idx in range(self.aug_env.num_envs)])
                    switch_goal_flag = [False for _ in range(self.aug_env.num_envs)]
                    env_end_flag = [False for _ in range(self.aug_env.num_envs)]
                    env_end_step = [np.inf for _ in range(self.aug_env.num_envs)]
                    env_restart_state = self.restart_states[:self.aug_env.num_envs]
                    self.aug_env.env_method('set_state', env_restart_state)
                    if 'FetchStack' in self.env_id:
                        self.aug_env.env_method('set_current_nobject', self.current_nobject[:self.aug_env.num_envs])
                        self.aug_env.env_method('set_selected_objects', self.selected_objects[:self.aug_env.num_envs])
                        # Use pick and place as sub-task
                        self.aug_env.env_method('set_task_mode', np.zeros(self.aug_env.num_envs))
                    env_obs = self.aug_env.env_method('get_obs')
                    env_obs = [convert_dict_to_obs(d) for d in env_obs]
                    increment_step = 0
                    while not sum(env_end_flag) == self.aug_env.num_envs:
                        # Switch subgoal according to switch_goal_flag, and update observation
                        switch_subgoal(switch_goal_flag, env_obs)
                        env_action, _ = self.predict(np.array(env_obs))
                        if 'FetchStack' in self.env_id:
                            relabel_env_obs = self.aug_env.env_method('switch_obs_goal', env_obs, ultimate_goals,
                                                                      self.task_mode)
                        else:
                            relabel_env_obs = self.aug_env.env_method('switch_obs_goal', env_obs, ultimate_goals)
                        clipped_actions = env_action
                        # Clip the actions to avoid out of bound error
                        if isinstance(self.aug_env.action_space, gym.spaces.Box):
                            clipped_actions = np.clip(env_action, self.aug_env.action_space.low,
                                                      self.aug_env.action_space.high)
                        env_next_obs, _, _, env_info = self.aug_env.step(clipped_actions)
                        self.num_aug_steps += (self.aug_env.num_envs - sum(env_end_flag))
                        if self.reward_type == 'sparse':
                            temp_info = [None for _ in range(self.aug_env.num_envs)]
                        else:
                            temp_info = [{'previous_obs': env_obs[i]} for i in range(self.aug_env.num_envs)]
                        if 'FetchStack' in self.env_id:
                            _retask_env_next_obs = env_next_obs.copy()
                            _retask_env_next_obs[:, self.obs_dim - 2:self.obs_dim] = 0
                            _retask_env_next_obs[:, self.obs_dim - 1] = 1  # Stack
                            relabel_env_next_obs = self.aug_env.env_method('switch_obs_goal', env_next_obs, ultimate_goals, self.task_mode)
                            env_reward = self.aug_env.env_method('compute_reward', _retask_env_next_obs, ultimate_goals,
                                                                 temp_info)
                            if self.reward_type != "sparse":
                                env_reward_and_success = self.aug_env.env_method('compute_reward_and_success',
                                                                                 _retask_env_next_obs, ultimate_goals,
                                                                                 temp_info)
                        else:
                            relabel_env_next_obs = self.aug_env.env_method('switch_obs_goal', env_next_obs,
                                                                           ultimate_goals)
                            env_reward = self.aug_env.env_method('compute_reward', env_next_obs, ultimate_goals,
                                                                 temp_info)
                            if self.reward_type != "sparse":
                                env_reward_and_success = self.aug_env.env_method('compute_reward_and_success',
                                                                                 env_next_obs, ultimate_goals,
                                                                                 temp_info)
                        for idx in range(self.aug_env.num_envs):
                            # obs, act, reward, next_obs, done
                            env_increment_storage[idx].append(
                                (relabel_env_obs[idx], env_action[idx], env_reward[idx], relabel_env_next_obs[idx], False))
                            # if idx == 0:
                            #     print(increment_step, env_obs[idx][:9], env_next_obs[idx][:9], env_reward[idx])
                        env_obs = env_next_obs
                        increment_step += 1
                        # print('increment step', increment_step)

                        for idx, info in enumerate(env_info):
                            # Special case, the agent succeeds the final goal half way
                            if self.reward_type == 'sparse' and env_reward[idx] > 0 and env_end_flag[idx] is False:
                                env_end_flag[idx] = True
                                env_end_step[idx] = env_restart_steps[idx] + increment_step
                            elif self.reward_type != 'sparse' and env_reward_and_success[idx][1] and env_end_flag[
                                idx] is False:
                                env_end_flag[idx] = True
                                env_end_step[idx] = env_restart_steps[idx] + increment_step
                            # Exceed time limit
                            if env_end_flag[idx] is False and env_restart_steps[idx] + increment_step \
                                    > self.get_horizon(self.current_nobject[idx] if 'FetchStack' in self.env_id else None):
                                env_end_flag[idx] = True
                                # But env_end_step is still np.inf
                            if info['is_success']:
                                if len(env_subgoals[idx]) >= 2:
                                    switch_goal_flag[idx] = True
                                    # if idx == 0:
                                    #     print('switch goal')
                                elif env_end_flag[idx] == False:
                                    # this is the end
                                    env_end_flag[idx] = True
                                    env_end_step[idx] = env_restart_steps[idx] + increment_step
                                else:
                                    pass
                        if increment_step >= self.get_horizon(max(self.current_nobject[:self.aug_env.num_envs])
                                                              if 'FetchStack' in self.env_id else None) \
                                - min(env_restart_steps):
                            break

                    # print('end step', env_end_step)
                    for idx, end_step in enumerate(env_end_step):
                        if end_step <= self.get_horizon(self.current_nobject[idx] if 'FetchStack' in self.env_id else None):
                            # log_debug_value(self.debug_value1[idx], self.debug_value2[idx], np.argmax(temp_subgoals[idx][3:]), True)
                            # print(temp_subgoals[idx])
                            # is_self_aug = temp_subgoals[idx][3]
                            transitions = env_increment_storage[idx][:end_step - env_restart_steps[idx]]
                            for i in range(len(env_storage[idx])):
                                if isinstance(self.augment_replay_buffer.replay_buffer, MultiWorkerReplayBuffer) \
                                        or isinstance(self.augment_replay_buffer.replay_buffer, PrioritizedMultiWorkerReplayBuffer):
                                    self.augment_replay_buffer.add(
                                        *([np.expand_dims(item, axis=0) for item in env_storage[idx][i]]))
                                else:
                                    self.augment_replay_buffer.add(*(env_storage[idx][i]))
                            for i in range(len(transitions)):
                                if i == len(transitions) - 1:
                                    temp = list(transitions[i])
                                    temp[-1] = True
                                    transitions[i] = tuple(temp)
                                if isinstance(self.augment_replay_buffer.replay_buffer, MultiWorkerReplayBuffer) \
                                        or isinstance(self.augment_replay_buffer.replay_buffer, PrioritizedMultiWorkerReplayBuffer):
                                    self.augment_replay_buffer.add(
                                        *([np.expand_dims(item, axis=0) for item in transitions[i]]))
                                else:
                                    self.augment_replay_buffer.add(*(transitions[i]))
                        # else:
                        #     log_debug_value(self.debug_value1[idx], self.debug_value2[idx], np.argmax(temp_subgoals[idx][3:]), False)


                    self.restart_steps = self.restart_steps[self.aug_env.num_envs:]
                    self.subgoals = self.subgoals[self.aug_env.num_envs:]
                    self.restart_states = self.restart_states[self.aug_env.num_envs:]
                    self.transition_storage = self.transition_storage[self.aug_env.num_envs:]
                    if 'FetchStack' in self.env_id:
                        self.current_nobject = self.current_nobject[self.aug_env.num_envs:]
                        self.selected_objects = self.selected_objects[self.aug_env.num_envs:]
                        self.task_mode = self.task_mode[self.aug_env.num_envs:]

                if step % self.train_freq == 0:
                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                           or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._train_step(step, writer, current_lr))
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                if len(episode_rewards[0][-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(
                        np.mean(np.concatenate([episode_rewards[i][-101:-1] for i in range(self.env.env.num_envs)]))),
                                        1)

                num_episodes = sum([len(episode_rewards[i]) for i in range(len(episode_rewards))])
                self.num_timesteps += self.env.env.num_envs
                # Display training infos
                if self.verbose >= 1 and dones[0] and log_interval is not None and len(episode_rewards[0]) % (log_interval // self.env.env.num_envs) == 0:
                    fps = int(self.num_timesteps / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    logger.logkv('select subgoal time', select_subgoal_time)
                    if len(episode_successes[0]) > 0:
                        logger.logkv("success rate", np.mean(np.concatenate([episode_successes[i][-100:] for i in range(self.env.env.num_envs)])))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv('augmented steps', len(self.augment_replay_buffer))
                    logger.logkv("original_timesteps", self.num_timesteps)
                    logger.logkv("total timesteps", self.num_timesteps + self.num_aug_steps)
                    logger.logkv("random_ratio", self.env.env.get_attr('random_ratio')[0])
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            return self

    def get_horizon(self, current_nobject):
        if 'FetchStack' in self.env_id and self.sequential:
            return max(current_nobject * 50, 100)
        return self.env.env.get_attr('spec')[0].max_episode_steps

    def select_subgoal(self, transition_buf, k, tower_height=None):
        if 'FetchStack' in self.env_id:
            assert tower_height is not None
        obs_buf, *_ = zip(*transition_buf)
        obs_buf = np.asarray(obs_buf)
        if 'FetchStack' in self.env_id and tower_height[-1] + 0.05 - obs_buf[-1][self.obs_dim + self.goal_dim + 2] > 0.01:
            # Tower height is equal to (or higher than) goal height.
            # print('towerheight exceed goalheight')
            return np.array([]), np.array([])
        sample_t = np.random.randint(0, len(transition_buf), 4096)
        sample_obs = obs_buf[sample_t]
        noise = np.random.uniform(low=-self.noise_mag, high=self.noise_mag, size=(len(sample_t), 2))
        sample_obs_buf = []
        subgoal_obs_buf = []
        filter_low = 0.5
        filter_high = 1.0
        if 'FetchPushWallobstacle' in self.env_id:
            filter_low = 0.7
            filter_high = 0.9 if self.env.env.get_attr('random_ratio')[0] < 1 else 1.0
        elif 'MasspointMaze' in self.env_id:
            filter_low = 0.7
            filter_high = 0.9
        if 'FetchStack' in self.env_id:
            ultimate_idx = np.argmax(sample_obs[0][self.obs_dim + self.goal_dim + 3:])
            filter_subgoal = True
            sample_height = np.array(tower_height)[sample_t]
            for object_idx in range(0, self.n_object):
                if np.linalg.norm(sample_obs[0][3 + object_idx * 3: 3 + (object_idx + 1) * 3]) < 1e-3:
                    # This object is masked
                    continue
                obstacle_xy = sample_obs[:, 3 * (object_idx + 1):3 * (object_idx + 1) + 2] + noise
                # Find how many objects have been stacked
                obstacle_height = np.expand_dims(sample_height + 0.05, axis=1)
                # obstacle_height = max(sample_obs[0][self.obs_dim + self.goal_dim + 2] - 0.05, 0.425) * np.ones((len(sample_t), 1))
                obstacle_xy = np.concatenate([obstacle_xy, obstacle_height], axis=-1)
                sample_obs[:, 3 * (object_idx + 1):3 * (object_idx + 1) + 3] = obstacle_xy
                sample_obs[:, 3 * (object_idx + 1 + self.n_object):3 * (object_idx + 1 + self.n_object) + 3] \
                    = sample_obs[:, 3 * (object_idx + 1):3 * (object_idx + 1) + 3] - sample_obs[:, 0:3]
                sample_obs[:, self.obs_dim:self.obs_dim + 3] = sample_obs[:,
                                                               3 * (ultimate_idx + 1):3 * (ultimate_idx + 1) + 3]
                sample_obs_buf.append(sample_obs.copy())

                subgoal_obs = obs_buf[sample_t]
                # if debug:
                #     subgoal_obs = np.tile(subgoal_obs, (2, 1))
                subgoal_obs[:, self.obs_dim - 2: self.obs_dim] = np.array([1, 0])  # Pick and place
                subgoal_obs[:, self.obs_dim:self.obs_dim + 3] = subgoal_obs[:,
                                                                3 * (object_idx + 1):3 * (object_idx + 1) + 3]
                one_hot = np.zeros(self.n_object)
                one_hot[object_idx] = 1
                subgoal_obs[:, self.obs_dim + 3:self.obs_dim + self.goal_dim] = one_hot
                subgoal_obs[:, self.obs_dim + self.goal_dim:self.obs_dim + self.goal_dim + 3] = obstacle_xy
                # subgoal_obs[:, self.obs_dim + self.goal_dim + 2:self.obs_dim + self.goal_dim + 3] = subgoal_obs[:, 3 * (
                # object_idx + 1) + 2:3 * (object_idx + 1) + 3]
                subgoal_obs[:, self.obs_dim + self.goal_dim + 3:self.obs_dim + self.goal_dim * 2] = one_hot
                subgoal_obs_buf.append(subgoal_obs)
        elif 'MasspointMaze' in self.env_id:
            filter_subgoal = True
            ego_xy = sample_obs[:, 0:2] + noise
            # Path 2
            sample_obs[:, 0: 2] = ego_xy
            sample_obs[:, self.obs_dim: self.obs_dim + 2] = ego_xy
            sample_obs_buf.append(sample_obs.copy())
            # Path 1
            subgoal_obs = obs_buf[sample_t]
            subgoal_obs[:, self.obs_dim:self.obs_dim + 3] = subgoal_obs[:, 0:3]
            subgoal_obs[:, self.obs_dim + self.goal_dim:self.obs_dim + self.goal_dim + 2] = ego_xy
            if self.env_id != 'MasspointMaze-v3':
                subgoal_obs[:, self.obs_dim + self.goal_dim + 2:self.obs_dim + self.goal_dim + 3] = subgoal_obs[:, 2:3]
            subgoal_obs_buf.append(subgoal_obs)
        else:
            ultimate_idx = np.argmax(sample_obs[0][self.obs_dim + self.goal_dim + 3:])
            filter_subgoal = True
            for object_idx in range(self.n_object):  # Also search self position
                obstacle_xy = sample_obs[:, 3 * (object_idx+1):3*(object_idx+1) + 2] + noise
                # Path2
                if object_idx == self.n_object - 1 and self.env_id is 'MasspointPushDoubleObstacle-v2':
                    sample_obs[0: 2] = obstacle_xy
                sample_obs[:, 3*(object_idx+1):3*(object_idx+1)+2] = obstacle_xy
                sample_obs[:, 3*(object_idx+1+self.n_object):3*(object_idx+1+self.n_object)+2] \
                    = sample_obs[:, 3*(object_idx+1):3*(object_idx+1)+2] - sample_obs[:, 0:2]
                # achieved_goal
                sample_obs[:, self.obs_dim:self.obs_dim + 3] \
                    = sample_obs[:, 3 * (ultimate_idx + 1):3 * (ultimate_idx + 1) + 3]
                sample_obs_buf.append(sample_obs.copy())

                # Path1
                subgoal_obs = obs_buf[sample_t]
                # achieved_goal
                subgoal_obs[:, self.obs_dim:self.obs_dim+3] = subgoal_obs[:, 3*(object_idx+1):3*(object_idx+1)+3]
                one_hot = np.zeros(self.n_object)
                one_hot[object_idx] = 1
                subgoal_obs[:, self.obs_dim+3:self.obs_dim+self.goal_dim] = one_hot
                # desired_goal
                subgoal_obs[:, self.obs_dim+self.goal_dim:self.obs_dim+self.goal_dim+2] = obstacle_xy
                subgoal_obs[:, self.obs_dim+self.goal_dim+2:self.obs_dim+self.goal_dim+3] \
                    = subgoal_obs[:, 3*(object_idx+1)+2:3*(object_idx+1)+3]
                subgoal_obs[:, self.obs_dim+self.goal_dim+3:self.obs_dim+self.goal_dim*2] = one_hot
                subgoal_obs_buf.append(subgoal_obs)
        # print(len(sample_obs_buf))
        if len(sample_obs_buf) == 0:
            return np.array([]), np.array([])
        sample_obs_buf = np.concatenate(sample_obs_buf, axis=0)
        # value2 = self.model.value(sample_obs_buf)
        subgoal_obs_buf = np.concatenate(subgoal_obs_buf)
        # value1 = self.model.value(subgoal_obs_buf)

        # _values = self.model.value(np.concatenate([sample_obs_buf, subgoal_obs_buf], axis=0))
        feed_dict = {self.observations_ph: np.concatenate([sample_obs_buf, subgoal_obs_buf], axis=0)}
        _values = np.squeeze(self.sess.run(self.step_ops[6], feed_dict), axis=-1)
        value2 = _values[:sample_obs_buf.shape[0]]
        value1 = _values[sample_obs_buf.shape[0]:]
        normalize_value1 = (value1 - np.min(value1)) / (np.max(value1) - np.min(value1))
        normalize_value2 = (value2 - np.min(value2)) / (np.max(value2) - np.min(value2))
        # best_idx = np.argmax(normalize_value1 * normalize_value2)
        ind = np.argsort(normalize_value1 * normalize_value2)
        good_ind = ind[-k:]
        if filter_subgoal:
            mean_values = (value1[good_ind] + value2[good_ind]) / 2
            # print(mean_values)
            assert mean_values.shape[0] == k
            # Filter by hard threshold
            # In the beginning, the value fn tends to over estimate
            filtered_idx = np.where(np.logical_and(mean_values < filter_high, mean_values > filter_low))[0]
            good_ind = good_ind[filtered_idx]

        restart_step = sample_t[good_ind % len(sample_t)]
        subgoal = subgoal_obs_buf[good_ind, self.obs_dim + self.goal_dim:self.obs_dim + self.goal_dim * 2]

        return restart_step, subgoal

    def logpac(self, action):
        from stable_baselines.sac.policies import gaussian_likelihood, EPS
        act_mu = self.policy_tf.act_mu
        log_std = tf.log(self.policy_tf.std)
        # Potentially we need to clip atanh and pass gradient
        log_u = gaussian_likelihood(tf.atanh(tf.clip_by_value(action, -0.99, 0.99)), act_mu, log_std)
        log_ac = log_u - tf.reduce_sum(tf.log(1 - action ** 2 + EPS), axis=1)
        return log_ac

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have action probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and ouputed.")

        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
            "target_entropy": self.target_entropy,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
