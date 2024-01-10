import time
import sys
import multiprocessing
from collections import deque

import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, \
    tf_util  # ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
# from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger
from my_common import get_observertion_space, get_action_space
# from my_common import get_modify_act, get_prev2obs
# import random
from my_baselines import ActorCriticRLModel, SetVerbosity, TensorboardWriter
from my_common import total_rate_logger
from my_common.runners_v2 import AbstractEnvRunner
from my_common import HindSightBuffer
import copy


class PPO2(ActorCriticRLModel):
    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param cliprange_vf: (float or callable) Clipping parameter for the value function, it can be a function.
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling.
        To deactivate value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False):

        super(PPO2, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                   _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)

        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.cliprange_vf = cliprange_vf
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.graph = None
        self.sess = None
        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.clipfrac = None
        self.params = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.step = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.n_batch = None
        self.summary = None
        self.episode_reward = None

        ##  tu2id4n
        self.observation_space = get_observertion_space()
        self.action_space = get_action_space()
        self.old_params = []
        self.win_rate = None
        # self.pre_action_ph = tf.placeholder(dtype=tf.int32, shape=[None]+[], name='pre_action_ph')

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):
            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_steps

            n_cpu = multiprocessing.cpu_count()
            if sys.platform == 'darwin':
                n_cpu //= 2

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, " \
                                                                 "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch=None, reuse=False, old_params=self.old_params, **self.policy_kwargs)
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch=None,
                                              reuse=True, old_params=self.old_params, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    vpred = train_model.value_flat

                    # Value function clipping: not present in the original PPO
                    if self.cliprange_vf is None:
                        # Default behavior (legacy from OpenAI baselines):
                        # use the same clipping as for the policy
                        self.clip_range_vf_ph = self.clip_range_ph
                        self.cliprange_vf = self.cliprange
                    elif isinstance(self.cliprange_vf, (float, int)) and self.cliprange_vf < 0:
                        # Original PPO implementation: no value function clipping
                        self.clip_range_vf_ph = None
                    else:
                        # Last possible behavior: clipping range
                        # specific to the value function
                        self.clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_vf_ph")

                    if self.clip_range_vf_ph is None:
                        # No clipping
                        vpred_clipped = train_model.value_flat
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        vpred_clipped = self.old_vpred_ph + \
                                        tf.clip_by_value(train_model.value_flat - self.old_vpred_ph,
                                                         - self.clip_range_vf_ph, self.clip_range_vf_ph)

                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpred_clipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)
                    grads = tf.gradients(loss, self.params)
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))
                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    if self.clip_range_vf_ph is not None:
                        tf.summary.scalar('clip_range_vf', tf.reduce_mean(self.clip_range_vf_ph))

                    tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probabilty', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary = tf.summary.merge_all()

                # 输出计算图
                # writer = tf.summary.FileWriter('../logs/graph',self.graph)
                # writer.flush()
                # writer.close()

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
                    writer, states=None, cliprange_vf=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions,
                  self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True, save_interval=None, save_path=None, gamma=0.99, n_steps=128):
        print('----------------------------------------------')
        print('|                 L E A R N                  |')
        print('----------------------------------------------')

        print("num timesteps = " + str(int(total_timesteps / 1000000)) + 'm')
        # print("num_envs = ", self.num_envs)
        print("save_interval = " + str(int(save_interval / 1000)) + 'k')
        print()
        save_interval_st = save_interval
        self.gamma = gamma
        self.n_steps = n_steps
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()  # 去掉参数 seed ?

            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)
            hindsight_buffer = HindSightBuffer(self.n_steps, self.gamma, self.lam)
            self.episode_reward = np.zeros((self.n_envs,))
            self.win_rate = np.zeros((self.n_envs,))
            self.tie_rate = np.zeros((self.n_envs,))
            self.loss_rate = np.zeros((self.n_envs,))

            # ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()

            n_updates = total_timesteps // self.n_batch  # self.n_batch = self.n_envs(8) * self.n_steps(128)
            for update in range(1, n_updates + 1):
                assert self.n_batch % self.nminibatches == 0  # self.nminibatches == 4
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)
                # true_reward is the reward without discount
                obs, returns, masks, actions, values, neglogpacs, states, true_reward, \
                win_rates, tie_rates, loss_rates, obs_nf = runner.run()
                self.num_timesteps += self.n_batch
                # ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
                                                                            self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, writer=writer,
                                                                 update=timestep, cliprange_vf=cliprange_vf_now))
                else:  # recurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_envs + epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, update=timestep,
                                                                 writer=writer, states=mb_states,
                                                                 cliprange_vf=cliprange_vf_now))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                      true_reward.reshape((self.n_envs, 2*self.n_steps)),
                                                                      masks.reshape((self.n_envs, 2*self.n_steps)),
                                                                      writer, self.num_timesteps)
                    self.win_rate = total_rate_logger(self.win_rate,
                                                      win_rates.reshape((self.n_envs, self.n_steps)),
                                                      masks[:5120].reshape((self.n_envs, self.n_steps)),
                                                      writer, self.num_timesteps,
                                                      name='win_rate')
                    self.tie_rate = total_rate_logger(self.tie_rate,
                                                      tie_rates.reshape((self.n_envs, self.n_steps)),
                                                      masks[:5120].reshape((self.n_envs, self.n_steps)),
                                                      writer, self.num_timesteps,
                                                      name='tie_rate')
                    self.loss_rate = total_rate_logger(self.loss_rate,
                                                       loss_rates.reshape((self.n_envs, self.n_steps)),
                                                       masks[:5120].reshape((self.n_envs, self.n_steps)),
                                                       writer, self.num_timesteps,
                                                       name='loss_rate')

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    # if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                    #     logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                    #     logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # save interval
                if self.num_timesteps >= save_interval_st:
                    save_interval_st += save_interval
                    s_path = save_path + '_' + str(int(self.num_timesteps / 10000)) + 'k.zip'
                    self.save(save_path=s_path)

            return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "cliprange_vf": self.cliprange_vf,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "old_params": self.old_params
        }

        params_to_save = self.get_parameters()

        print('----------------------------------------------')
        print('|                  S A V E                   |')
        print('----------------------------------------------')
        print('load_path =', save_path)
        print("num of current networks = ", len(self.old_params))
        print("len_parm = ", len(params_to_save))
        print()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, using_pgn=False, tensorboard_log=None, **kwargs):
        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)
        print('----------------------------------------------')
        print('|                  L O A D                   |')
        print('----------------------------------------------')

        print('load_path =', load_path)
        print('using pgn = : ', using_pgn)
        print('tensorboard_log = ', tensorboard_log)
        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))
        model = cls(policy=data["policy"], env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.tensorboard_log = tensorboard_log
        model.setup_model()
        model.load_parameters(params)

        # PGN MOD: Use new policy
        print("using_pgn = ", using_pgn)
        if using_pgn:
            len_parm = len(model.get_parameters())
            params_to_old = model.get_parameters()
            old = {}
            for _ in range(len_parm):
                key, val = params_to_old.popitem()
                key = key[6:-2]
                print(key)
                old[key] = val
                # print(key,val.shape)
            model.old_params.append(old)
            # print(model.old_params)
            print("**************** Save the old learned params")
            print("num of old networks = ", len(model.old_params))
            print("len_parm = ", len_parm)
            print()
            if env is not None:
                model.action_space = get_action_space()
            model.setup_model()

        return model

    def pretrain(self, dataset, n_epochs=10, learning_rate=1e-4,
                 adam_epsilon=1e-8, save_path=None):
        """
        Pretrain a model using behavior cloning:
        supervised learning given an expert dataset.

        NOTE: only Box and Discrete spaces are supported for now.

        :param dataset: (ExpertDataset) Dataset manager
        :param n_epochs: (int) Number of iterations on the training set
        :param learning_rate: (float) Learning rate
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param val_interval: (int) Report training and validation losses every n epochs.
            By default, every 10th of the maximum number of epochs.
        :return: (BaseRLModel) the pretrained model
        """

        print('----------------------------------------------')
        print('|               P R E T A I N                |')
        print('----------------------------------------------')

        print("n_epochs = ", n_epochs)
        print("save_path = ", save_path)
        continuous_actions = isinstance(self.action_space, gym.spaces.Box)
        discrete_actions = isinstance(self.action_space, gym.spaces.Discrete)

        assert discrete_actions or continuous_actions, 'Only Discrete and Box action spaces are supported'

        # Validate the model every 10% of the total number of iteration

        with self.graph.as_default():
            with tf.variable_scope('pretrain'):
                if continuous_actions:
                    obs_ph, actions_ph, deterministic_actions_ph = self._get_pretrain_placeholders()
                    loss = tf.reduce_mean(tf.square(actions_ph - deterministic_actions_ph))
                else:
                    obs_ph, actions_ph, actions_logits_ph = self._get_pretrain_placeholders()
                    # actions_ph has a shape if (n_batch,), we reshape it to (n_batch, 1)
                    # so no additional changes is needed in the dataloader
                    actions_ph = tf.expand_dims(actions_ph, axis=1)
                    one_hot_actions = tf.one_hot(actions_ph, self.action_space.n)
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=actions_logits_ph,
                        labels=tf.stop_gradient(one_hot_actions)
                    )
                    loss = tf.reduce_mean(loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon)
                optim_op = optimizer.minimize(loss, var_list=self.params)

            self.sess.run(tf.global_variables_initializer())

        if self.verbose > 0:
            print("Pretraining with Behavior Cloning...")

        for epoch_idx in range(int(n_epochs)):
            train_loss = 0.0
            # Full pass on the training set
            for _ in range(len(dataset.train_loader)):
                expert_obs, expert_actions = dataset.get_next_batch('train')
                feed_dict = {
                    obs_ph: expert_obs,
                    actions_ph: expert_actions,
                }
                train_loss_, _ = self.sess.run([loss, optim_op], feed_dict)
                train_loss += train_loss_

            train_loss /= len(dataset.train_loader)

            if self.verbose > 0 and (epoch_idx + 1) % 1 == 0:
                val_loss = 0.0
                # Full pass on the validation set
                for _ in range(len(dataset.val_loader)):
                    expert_obs, expert_actions = dataset.get_next_batch('val')
                    val_loss_, = self.sess.run([loss], {obs_ph: expert_obs,
                                                        actions_ph: expert_actions})
                    val_loss += val_loss_

                val_loss /= len(dataset.val_loader)
                if self.verbose > 0:
                    print("==== Training progress {:.2f}% ====".format(100 * (epoch_idx + 1) / n_epochs))
                    print('Epoch {}'.format(epoch_idx + 1))
                    print("Training loss: {:.6f}, Validation loss: {:.6f}".format(train_loss, val_loss))
                    print()

                print("Save pretrained model ", epoch_idx + 1)
                self.save(save_path + '_e' + str(epoch_idx + 1) + '.zip')
            # Free memory
            del expert_obs, expert_actions
        if self.verbose > 0:
            print("Pretraining done.")
        return self


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma

    def run(self, hindsight_buffer=None):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        mb_obs0, mb_rewards0, mb_actions0, mb_values0, mb_dones, mb_neglogpacs0, mb_obs_nf0 = \
            [], [], [], [], [], [], []
        mb_obs2, mb_rewards2, mb_actions2, mb_values2, mb_neglogpacs2, mb_obs_nf2 = \
            [], [], [], [], [], []
        win_rates, tie_rates, loss_rates, dead_flags0, dead_flags2 = [], [], [], [], []
        mb_states = self.states
        for _ in range(self.n_steps):
            actions0, values0, self.states, neglogpacs0 = self.model.step(self.obs0, self.states, self.dones)
            actions2, values2, self.states, neglogpacs2 = self.model.step(self.obs2, self.states, self.dones)
            mb_obs0.append(self.obs0.copy())
            mb_actions0.append(actions0)
            mb_values0.append(values0)
            mb_neglogpacs0.append(neglogpacs0)
            mb_dones.append(self.dones)
            mb_obs_nf0.append(copy.deepcopy(self.obs_nf0))

            mb_obs2.append(self.obs2.copy())
            mb_actions2.append(actions2)
            mb_values2.append(values2)
            mb_neglogpacs2.append(neglogpacs2)
            mb_obs_nf2.append(copy.deepcopy(self.obs_nf2))

            clipped_actions = zip(actions0, actions2)
            self.obs0[:], rewards0, self.obs_nf0[:], self.obs2[:], rewards2, self.obs_nf2[:], \
            self.dones, dead_flag0, dead_flag2, win_rate, tie_rate, loss_rate = self.env.step(clipped_actions)

            dead_flags0.append(dead_flag0)
            dead_flags2.append(dead_flag2)
            win_rates.append(win_rate)
            tie_rates.append(tie_rate)
            loss_rates.append(loss_rate)
            mb_rewards0.append(rewards0)
            mb_rewards2.append(rewards2)
        # batch of steps to batch of rollouts
        mb_obs0 = np.asarray(mb_obs0, dtype=self.obs0.dtype)
        mb_obs_nf0 = np.array(mb_obs_nf0)
        mb_rewards0 = np.asarray(mb_rewards0, dtype=np.float32)
        mb_actions0 = np.asarray(mb_actions0)
        mb_values0 = np.asarray(mb_values0, dtype=np.float32)
        mb_neglogpacs0 = np.asarray(mb_neglogpacs0, dtype=np.float32)
        last_values0 = self.model.value(self.obs0, self.states, self.dones)

        mb_obs2 = np.asarray(mb_obs2, dtype=self.obs2.dtype)
        mb_obs_nf2 = np.array(mb_obs_nf2)
        mb_rewards2 = np.asarray(mb_rewards2, dtype=np.float32)
        mb_actions2 = np.asarray(mb_actions2)
        mb_values2 = np.asarray(mb_values2, dtype=np.float32)
        mb_neglogpacs2 = np.asarray(mb_neglogpacs2, dtype=np.float32)
        last_values2 = self.model.value(self.obs2, self.states, self.dones)

        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        win_rates = np.array(win_rates)
        tie_rates = np.array(tie_rates)
        loss_rates = np.array(loss_rates)
        dead_flags0 = np.array(dead_flags0)
        dead_flags2 = np.array(dead_flags2)
        true_reward0 = np.copy(mb_rewards0)
        true_reward2 = np.copy(mb_rewards2)
        # last_values0 = np.array(last_values0)
        # last_values2 = np.array(last_values2)
        true_reward = np.concatenate((true_reward0, true_reward2))

        mb_obs = np.concatenate((mb_obs0, mb_obs2))
        mb_actions = np.concatenate((mb_actions0, mb_actions2))
        mb_values = np.concatenate((mb_values0, mb_values2))
        mb_neglogpacs = np.concatenate((mb_neglogpacs0, mb_neglogpacs2))
        mb_rewards = np.concatenate((mb_rewards0, mb_rewards2))
        mb_obs_nf = np.concatenate((mb_obs_nf0, mb_obs_nf2))
        dead_flags = np.concatenate((dead_flags0, dead_flags2))
        mb_dones = np.concatenate((mb_dones, mb_dones))
        last_values = np.concatenate((last_values0, last_values2))
        last_dones = np.concatenate((self.dones, self.dones))

        # print('obs', mb_obs.shape)
        # print('actions', mb_actions.shape)
        # print('values', mb_values.shape)
        # print('rewards', true_reward.shape)
        # print('dones', mb_dones.shape)
        # print('last_values', last_values.shape)



        mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_rewards, mb_obs_nf, \
        mb_dones, dead_flags, win_rates, tie_rates, loss_rates, true_reward = \
            map(swap_and_flatten,
                (mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_rewards, mb_obs_nf,
                 mb_dones, dead_flags, win_rates, tie_rates, loss_rates, true_reward))


        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        # ep_infos = np.array(ep_infos)
        count = 0
        mb_obs = list(mb_obs)
        for step in reversed(range(len(mb_rewards))):
            if (step + 1) % 128 == 0:
                last_gae_lam = 0
                count += 1
                nextnonterminal = 1.0 - last_dones[-count]
                nextvalues = last_values[-count]
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            if dead_flags[step] == 1 and mb_dones[step] != 1 and step+1 < len(mb_rewards):
                mb_advs = np.delete(mb_advs, step+1)
                mb_actions = np.delete(mb_actions, step+1)
                mb_values = np.delete(mb_values, step+1)
                mb_neglogpacs = np.delete(mb_neglogpacs, step+1)
                mb_rewards = np.delete(mb_rewards, step+1)
                mb_obs.pop(step+1)
                true_reward[step] = 0

                continue
            # ∆ = r + 𝛄 * v' - v
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            # adv = ∆ + 𝛄 * lam * adv-pre
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs = np.array(mb_obs)
        print('obs', mb_obs.shape)
        print('actions', mb_actions.shape)
        print('values', mb_values.shape)
        print('rewards', true_reward.shape)
        print('dones', mb_dones.shape)
        print('last_values', last_values.shape)
        print('return', mb_returns.shape)

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, true_reward, \
               win_rates, tie_rates, loss_rates, mb_obs_nf


def get_schedule_fn(value_schedule):
    """
    Transform (if needed) learning rate and clip range
    to callable.

    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constfn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def constfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (function)
    """

    def func(_):
        return val

    return func


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)
