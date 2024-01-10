import time
import sys
import multiprocessing
from collections import deque

import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger
from utils.eval_stack import pp_eval_model


class PPO2_SIR(ActorCriticRLModel):
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

    def __init__(self, policy, env, aug_env=None, eval_env=None, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4,
                 vf_coef=0.5, aug_clip=0.1, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2,
                 cliprange_vf=None, n_candidate=4, dim_candidate=2, parallel=False, reuse_times=1, start_augment=0,
                 horizon=100, aug_adv_weight=1.0, curriculum=False, self_imitate=False, sil_clip=0.2, log_trace=False,
                 verbose=0, tensorboard_log=None, _init_setup_model=True,
                 policy_kwargs=None, full_tensorboard_log=False):

        super(PPO2_SIR, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                       _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)

        self.aug_env = aug_env
        self.eval_env = eval_env
        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.cliprange_vf = cliprange_vf
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.aug_clip = aug_clip
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.n_candidate = n_candidate
        self.dim_candidate = dim_candidate
        self.parallel = parallel
        self.start_augment = start_augment
        self.curriculum = curriculum
        self.self_imitate = self_imitate
        self.sil_clip = sil_clip
        self.log_trace = log_trace
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

        self.reuse_times = reuse_times
        self.aug_obs = []
        self.aug_act = []
        self.aug_neglogp = []
        self.aug_return = []
        self.aug_value = []
        self.aug_done = []
        self.aug_reward = []
        self.is_selfaug = []
        self.num_aug_steps = 0  # every interaction with simulator should be counted
        self.horizon = horizon
        self.aug_adv_weight = aug_adv_weight

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
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
                        "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, reuse=False, **self.policy_kwargs)
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)
                    train_aug_model = self.policy(self.sess, self.observation_space, self.action_space,
                                                  self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                                  reuse=True, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                    self.aug_action_ph = train_aug_model.pdtype.sample_placeholder([None], name="aug_action_ph")
                    self.aug_advs_ph = tf.placeholder(tf.float32, [None], name="aug_advs_ph")
                    self.aug_old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="aug_old_neglog_pac_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    aug_neglogpac = train_aug_model.proba_distribution.neglogp(self.aug_action_ph)
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
                    if self.self_imitate:
                        if not 'MasspointPush' in self.env.get_attr('spec')[0].id:
                            ratio = tf.exp(tf.minimum(self.old_neglog_pac_ph, 100) - tf.minimum(neglogpac, 100))
                        else:
                            ratio = tf.exp(tf.minimum(self.old_neglog_pac_ph, 20) - tf.minimum(neglogpac, 20))
                    else:
                        if 'MasspointPushMultiObstacle' in self.env.get_attr('spec')[0].id:
                            ratio = tf.exp(tf.minimum(self.old_neglog_pac_ph, 20) - neglogpac)
                            # ratio = tf.exp(tf.minimum(self.old_neglog_pac_ph - neglogpac, 10))
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    # if True:
                    #     pg_losses = -tf.clip_by_value(self.advs_ph, -1., 1.) * ratio
                    #     pg_losses2 = -tf.clip_by_value(self.advs_ph, -1., 1.) * tf.clip_by_value(
                    #         ratio, 1.0 - self.clip_range_ph, 1.0 + self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))
                    self.ratio_max = tf.reduce_max(ratio)
                    aug_ratio = tf.exp(self.aug_old_neglog_pac_ph - aug_neglogpac)
                    aug_pg_losses = -self.aug_advs_ph * aug_ratio
                    aug_pg_losses2 = -self.aug_advs_ph * tf.clip_by_value(aug_ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                          self.clip_range_ph)
                    self.aug_pg_loss = tf.reduce_mean(tf.maximum(aug_pg_losses, aug_pg_losses2))
                    self.aug_clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(aug_ratio - 1.0),
                                                                          self.clip_range_ph), tf.float32))
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef
                    # loss += self.aug_coef * self.aug_pg_loss

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)

                    # print(tf.trainable_variables())
                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        # print(self.params)
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)
                    grads = tf.gradients(loss, self.params)
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))
                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'advs_min', 'advs_max', 'ratio_max', 'max_neglogp', 'max_neglogp_origin', 'mean_neglogp']

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
                self.train_aug_model = train_aug_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                self.aug_neglogpac_op = aug_neglogpac
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary = tf.summary.merge_all()

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, is_demo, update,
                    writer, states=None, cliprange_vf=None, aug_obs_slice=None, aug_act_slice=None,
                    aug_neglog_pac_slice=None, aug_adv_slice=None):
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
        if not 'MasspointPush' in self.env.get_attr('spec')[0].id:
            advs = is_demo * np.clip(advs, self.aug_clip, np.inf) * self.aug_adv_weight + (1 - is_demo) * advs
        else:
            advs = is_demo * np.clip(advs, 0., 1.) * self.aug_adv_weight + (1 - is_demo) * advs
        # for i in range(advs.shape[0]):
        #     if is_demo[i]:
        #         if not 'MasspointPush' in self.env.get_attr('spec')[0].id:
        #             advs[i] = np.max([advs[i], self.aug_clip]) * self.aug_adv_weight
        #         else:
        #             advs[i] = np.clip(advs[i], 0., 1.) * self.aug_adv_weight
        if aug_adv_slice is not None:
            aug_adv_slice = (aug_adv_slice - aug_adv_slice.mean()) / (aug_adv_slice.std() + 1e-8)
        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions,
                  self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if aug_obs_slice is not None:
            td_map[self.train_aug_model.obs_ph] = aug_obs_slice
            td_map[self.aug_action_ph] = aug_act_slice
            # td_map[self.aug_advs_ph] = np.max(advs) * np.ones(advs.shape)
            td_map[self.aug_advs_ph] = aug_adv_slice
            # print('aug advs mean', np.mean(aug_adv_slice))
            td_map[self.aug_old_neglog_pac_ph] = aug_neglog_pac_slice
            # print('old aug neglog pac mean', np.mean(aug_neglog_pac_slice))
            _aug_neglog_pac = self.sess.run(self.aug_neglogpac_op, td_map)
            # print('aug neglog pac mean', np.mean(_aug_neglog_pac))
        # else:
        #     # td_map[self.train_aug_model.obs_ph] = np.zeros(obs.shape)
        #     # td_map[self.aug_action_ph] = np.zeros(actions.shape)
        #     td_map[self.train_aug_model.obs_ph] = obs
        #     td_map[self.aug_action_ph] = actions
        #     td_map[self.aug_advs_ph] = np.zeros(advs.shape)
        #     # td_map[self.aug_old_neglog_pac_ph] = np.zeros(neglogpacs.shape)
        #     td_map[self.aug_old_neglog_pac_ph] = neglogpacs

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
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, ratio_max, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self.ratio_max, self._train], td_map)
            # if aug_obs_slice is not None:
            #     print('demo loss', demo_loss)
                # exit()
        if len(np.where(is_demo < 0.5)[0]):
            original_maxneglogp = np.max(neglogpacs[np.where(is_demo < 0.5)[0]])
        else:
            original_maxneglogp = 0.

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac, np.min(advs), np.max(advs), ratio_max, np.max(neglogpacs), original_maxneglogp, np.mean(neglogpacs)

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            if not self.parallel:
                runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam,
                                aug_env=self.aug_env, n_candidate=self.n_candidate)
            elif self.self_imitate:
                from baselines.ppo_sir.sil_runner import SILRunner
                runner = SILRunner(env=self.env, aug_env=self.aug_env, model=self, n_steps=self.n_steps,
                                                 gamma=self.gamma, lam=self.lam, n_candidate=self.n_candidate,
                                                 dim_candidate=self.dim_candidate, horizon=self.horizon)
            else:
                if self.env.get_attr('n_object')[0] > 0:
                    if self.dim_candidate != 3:
                        from baselines.ppo_sir.sir_runner import SIRRunner
                        # runner = ParallelRunner(env=self.env, aug_env=self.aug_env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam,
                        #                         n_candidate=self.n_candidate, horizon=self.horizon)
                        runner = SIRRunner(env=self.env, aug_env=self.aug_env, model=self, n_steps=self.n_steps,
                                           gamma=self.gamma, lam=self.lam, n_candidate=self.n_candidate,
                                           dim_candidate=self.dim_candidate,
                                           horizon=self.horizon, log_trace=self.log_trace)
                    else:
                        # Stacking.
                        from baselines.ppo_sir.sir_runner_stack import SIRRunner
                        runner = SIRRunner(env=self.env, aug_env=self.aug_env, model=self, n_steps=self.n_steps,
                                           gamma=self.gamma, lam=self.lam, n_candidate=self.n_candidate,
                                           dim_candidate=self.dim_candidate,
                                           horizon=self.horizon)
                else:
                    # Maze.
                    from baselines.ppo_sir.sir_runner_maze import SIRRunner
                    runner = SIRRunner(env=self.env, aug_env=self.aug_env, model=self, n_steps=self.n_steps,
                                       gamma=self.gamma, lam=self.lam, n_candidate=self.n_candidate,
                                       dim_candidate=self.dim_candidate, horizon=self.horizon)
            self.episode_reward = np.zeros((self.n_envs,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()

            n_updates = total_timesteps // self.n_batch
            total_success = 0
            original_success = 0
            _reuse_times = self.reuse_times
            start_decay = n_updates
            pp_sr_buf = deque(maxlen=5)
            for update in range(1, n_updates + 1):
                assert self.n_batch % self.nminibatches == 0
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)
                if self.curriculum:
                    if 'FetchStack' in self.env.get_attr('spec')[0].id:
                        # Stacking
                        pp_sr = pp_eval_model(self.eval_env, self)
                        pp_sr_buf.append(pp_sr)
                        print('Pick-and-place success rate', np.mean(pp_sr_buf))
                        if start_decay == n_updates and np.mean(pp_sr_buf) > 0.8:
                            start_decay = update
                        _ratio = np.clip(0.7 - 0.8 * (update - start_decay) / 380, 0.3, 0.7) # from 0.7 to 0.3
                    elif 'FetchPushWallObstacle' in self.env.get_attr('spec')[0].id:
                        _ratio = max(1.0 - (update - 1.0) / n_updates, 0.0)
                    else:
                        raise NotImplementedError
                    self.env.env_method('set_random_ratio', _ratio)
                    print('Set random_ratio to', self.env.get_attr('random_ratio')[0])
                aug_success_ratio = (total_success - original_success) / (total_success + 1e-8)
                if aug_success_ratio > 1.0:
                # if aug_success_ratio > 0.25:
                    _reuse_times -= 1
                    _reuse_times = max(1, _reuse_times)
                else:
                    _reuse_times = min(self.reuse_times, _reuse_times + 1)
                # Reuse goalidx=0 only once
                # if len(self.aug_obs) and (self.aug_obs[-1] is not None):
                #     other_aug_idx = np.where(self.is_selfaug[-1] < 0.5)[0]
                #     self.aug_obs[-1] = self.aug_obs[-1][other_aug_idx] if len(other_aug_idx) else None
                #     self.aug_act[-1] = self.aug_act[-1][other_aug_idx] if len(other_aug_idx) else None
                #     self.aug_neglogp[-1] = self.aug_neglogp[-1][other_aug_idx] if len(other_aug_idx) else None
                #     self.aug_return[-1] = self.aug_return[-1][other_aug_idx] if len(other_aug_idx) else None
                #     self.aug_value[-1] = self.aug_value[-1][other_aug_idx] if len(other_aug_idx) else None
                #     self.aug_done[-1] = self.aug_done[-1][other_aug_idx] if len(other_aug_idx) else None
                #     self.aug_reward[-1] = self.aug_reward[-1][other_aug_idx] if len(other_aug_idx) else None
                # Reuse other data
                # if _reuse_times > 1 and (not ('MasspointPush' in self.env.get_attr('spec')[0].id) or ('MasspointPush' in self.env.get_attr('spec')[0].id and update > 150)):
                # if _reuse_times > 1 and (not ('MasspointPush' in self.env.get_attr('spec')[0].id) or (
                #         'MasspointPush' in self.env.get_attr('spec')[0].id and update > 100)):
                if _reuse_times > 1:
                    self.aug_obs = self.aug_obs[-_reuse_times+1:] + [None]
                    self.aug_act = self.aug_act[-_reuse_times+1:] + [None]
                    self.aug_neglogp = self.aug_neglogp[-_reuse_times+1:] + [None]
                    self.aug_return = self.aug_return[-_reuse_times+1:] + [None]
                    self.aug_value = self.aug_value[-_reuse_times+1:] + [None]
                    self.aug_done = self.aug_done[-_reuse_times+1:] + [None]
                    self.aug_reward = self.aug_reward[-_reuse_times+1:] + [None]
                    self.is_selfaug = self.is_selfaug[-_reuse_times+1:] + [None]
                else:
                    self.aug_obs = [None]
                    self.aug_act = [None]
                    self.aug_neglogp = [None]
                    self.aug_return = [None]
                    self.aug_value = [None]
                    self.aug_done = [None]
                    self.aug_reward = [None]
                    self.is_selfaug = [None]
                # true_reward is the reward without discount
                temp_time0 = time.time()
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()
                is_demo = np.zeros(obs.shape[0])
                temp_time1 = time.time()
                print('runner.run() takes', temp_time1 - temp_time0)
                original_episodes = np.sum(masks)
                original_success = np.sum([info['r'] for info in ep_infos])
                total_success = original_success

                # augment_steps = 0 if self.aug_obs is None else self.aug_obs.shape[0]
                augment_steps = sum([item.shape[0] if item is not None else 0 for item in self.aug_obs])
                print([item.shape[0] if item is not None else 0 for item in self.aug_obs])
                # if self.aug_obs is not None:
                if augment_steps > 0:
                    if self.self_imitate and augment_steps / self.n_batch > self.sil_clip:
                        aug_sample_idx = np.random.randint(0, augment_steps, int(self.n_batch * self.sil_clip))
                    else:
                        aug_sample_idx = np.arange(augment_steps)
                    _aug_return = np.concatenate(list(filter(lambda v:v is not None, self.aug_return)), axis=0)
                    _aug_value = np.concatenate(list(filter(lambda v: v is not None, self.aug_value)), axis=0)
                    adv_clip_frac = np.sum((_aug_return - _aug_value) < (np.mean(returns - values) + self.aug_clip * np.std(returns - values))) / _aug_return.shape[0]
                    print('demo adv below average + %f std' % self.aug_clip, adv_clip_frac)
                    if self.self_imitate:
                        _aug_obs = np.concatenate(list(filter(lambda  v: v is not None, self.aug_obs)), axis=0)[aug_sample_idx]
                        obs = np.concatenate([obs, _aug_obs], axis=0)
                        _aug_return = _aug_return[aug_sample_idx]
                        returns = np.concatenate([returns, _aug_return], axis=0)
                        _aug_mask = np.concatenate(list(filter(lambda v: v is not None, self.aug_done)), axis=0)[aug_sample_idx]
                        masks = np.concatenate([masks, _aug_mask], axis=0)
                        _aug_action = np.concatenate(list(filter(lambda v: v is not None, self.aug_act)), axis=0)[aug_sample_idx]
                        actions = np.concatenate([actions, _aug_action], axis=0)
                        _aug_value = np.concatenate(list(filter(lambda v: v is not None, self.aug_value)), axis=0)[aug_sample_idx]
                        values = np.concatenate([values, _aug_value], axis=0)
                        _aug_neglogpac = np.concatenate(list(filter(lambda v: v is not None, self.aug_neglogp)), axis=0)[aug_sample_idx]
                        neglogpacs = np.concatenate([neglogpacs, _aug_neglogpac], axis=0)
                        is_demo = np.concatenate([is_demo, np.ones(len(aug_sample_idx))], axis=0)
                        _aug_reward = np.concatenate(list(filter(lambda v: v is not None, self.aug_reward)), axis=0)[aug_sample_idx]
                        total_success += np.sum(_aug_reward)
                        augment_steps = len(aug_sample_idx)
                    else:
                        obs = np.concatenate([obs, *(list(filter(lambda v:v is not None, self.aug_obs)))], axis=0)
                        returns = np.concatenate([returns, *(list(filter(lambda v:v is not None, self.aug_return)))], axis=0)
                        masks = np.concatenate([masks, *(list(filter(lambda v:v is not None, self.aug_done)))], axis=0)
                        actions = np.concatenate([actions, *(list(filter(lambda v:v is not None, self.aug_act)))], axis=0)
                        values = np.concatenate([values, *(list(filter(lambda v:v is not None, self.aug_value)))], axis=0)
                        neglogpacs = np.concatenate([neglogpacs, *(list(filter(lambda v:v is not None, self.aug_neglogp)))], axis=0)
                        is_demo = np.concatenate([is_demo, np.ones(augment_steps)], axis=0)
                        _aug_reward = np.concatenate(list(filter(lambda v:v is not None, self.aug_reward)), axis=0)
                        total_success += np.sum(_aug_reward)
                    print('augmented data length', obs.shape[0])
                self.num_timesteps += self.n_batch
                ep_info_buf.extend(ep_infos)
                total_episodes = np.sum(masks)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    recompute_neglogp_time = 0
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                    inds = np.arange(self.n_batch + augment_steps)
                    # print('length self.aug_obs', len(self.aug_obs), batch_size)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch + augment_steps, batch_size):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
                                                                            self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            _recompute_inds = np.where(is_demo[mbinds] > 0.5)[0]
                            recompute_neglogp_time0 = time.time()
                            if _recompute_inds.shape[0] > 0:
                                neglogpacs[_recompute_inds] = self.sess.run(
                                    self.aug_neglogpac_op, {self.train_aug_model.obs_ph: obs[_recompute_inds],
                                                            self.aug_action_ph: actions[_recompute_inds]})
                                if self.self_imitate:
                                    neglogpacs[_recompute_inds] = np.minimum(neglogpacs[_recompute_inds], 100)
                            recompute_neglogp_time += time.time() - recompute_neglogp_time0
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs, is_demo))
                            # if len(self.aug_obs) > batch_size:
                            #     aug_inds = np.random.choice(len(self.aug_obs), batch_size)
                            #     aug_obs_slice = np.array(self.aug_obs)[aug_inds]
                            #     aug_act_slice = np.array(self.aug_act)[aug_inds]
                            #     aug_neglog_pac_slice = np.array(self.aug_neglogp)[aug_inds]
                            #     aug_adv_slice = np.array(self.aug_adv)[aug_inds]
                            # else:
                            #     aug_obs_slice = None
                            #     aug_act_slice = None
                            #     aug_neglog_pac_slice = None
                            #     aug_adv_slice = None
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, writer=writer,
                                                                 update=timestep, cliprange_vf=cliprange_vf_now,
                                                                 ))
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

                print('recompute neglogp time', recompute_neglogp_time)
                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                      true_reward.reshape((self.n_envs, self.n_steps)),
                                                                      masks.reshape((self.n_envs, self.n_steps)),
                                                                      writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("original_timesteps", self.num_timesteps)
                    logger.logkv("total_timesteps", self.num_timesteps + self.num_aug_steps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.logkv("augment_steps", augment_steps)
                    # logger.logkv("original_success", original_success)
                    # logger.logkv("total_success", total_success)
                    logger.logkv("self_aug_ratio", np.mean(runner.self_aug_ratio))
                    logger.dumpkvs()

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

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
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam, aug_env, n_candidate):
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
        self.aug_env = aug_env
        self.n_candidate = n_candidate
        # obs param
        self.obs_dim = 40
        self.goal_dim = 5
        self.n_object = env.unwrapped.n_object
        # For restart
        self.ep_state_buf = [[] for _ in range(self.model.n_envs)]
        self.ep_transition_buf = [[] for _ in range(self.model.n_envs)]

    def run(self):
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
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []

        augment_data_time = 0
        for _ in range(self.n_steps):
            internal_states = self.env.env_method('get_state')
            for i in range(self.model.n_envs):
                self.ep_state_buf[i].append(internal_states[i])
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
            for i in range(self.model.n_envs):
                # self.ep_transition_buf[i].append((mb_obs[-1][i], mb_actions[-1][i], mb_values[-1][i],
                #                                   mb_neglogpacs[-1][i], mb_dones[-1][i], mb_rewards[-1][i]))
                self.ep_transition_buf[i].append((mb_obs[-1][i], mb_actions[-1][i], mb_values[-1][i],
                                                  mb_neglogpacs[-1][i], mb_dones[-1][i], mb_rewards[-1][i]))
            # _values = self.env.env_method('augment_data', self.ep_transition_buf, self.ep_state_buf)
            # print(_values)
            # exit()
            temp_time0 = time.time()
            for idx, done in enumerate(self.dones):
                if self.model.num_timesteps > self.model.start_augment and done:
                    # Check if this is failture
                    goal = self.ep_transition_buf[idx][0][0][-5:]
                    if np.argmax(goal[3:]) == 0 and (not infos[idx]['is_success']):
                        # Do augmentation
                        # Sample start step and perturbation
                        restart_steps, subgoals = self.select_subgoal(self.ep_transition_buf[idx], k=self.n_candidate)
                        # print('restart steps', restart_steps, 'subgoals', subgoals, 'ultimate goal', goal)
                        # augment_transition_buf = self.ep_transition_buf[idx][:restart_step]
                        for k in range(restart_steps.shape[0]):
                            restart_step = restart_steps[k]
                            subgoal = subgoals[k]
                            if restart_step > 0:
                                augment_obs_buf, augment_act_buf, augment_value_buf, \
                                augment_neglogp_buf, augment_done_buf, augment_reward_buf = \
                                    zip(*self.ep_transition_buf[idx][:restart_step])
                                augment_obs_buf = list(augment_obs_buf)
                                augment_act_buf = list(augment_act_buf)
                                augment_value_buf = list(augment_value_buf)
                                augment_neglogp_buf = list(augment_neglogp_buf)
                                augment_done_buf = list(augment_done_buf)
                                augment_reward_buf = list(augment_reward_buf)
                            else:
                                augment_obs_buf, augment_act_buf, augment_value_buf, \
                                augment_neglogp_buf, augment_done_buf, augment_reward_buf = [], [], [], [], [], []
                            augment_obs1, augment_act1, augment_value1, augment_neglogp1, augment_done1, augment_reward1, next_state = \
                                self.rollout_subtask(self.ep_state_buf[idx][restart_step], subgoal, len(augment_obs_buf), goal)
                            if augment_obs1 is not None:
                                # augment_transition_buf += augment_transition1
                                augment_obs_buf += augment_obs1
                                augment_act_buf += augment_act1
                                augment_value_buf += augment_value1
                                augment_neglogp_buf += augment_neglogp1
                                augment_done_buf += augment_done1
                                augment_reward_buf += augment_reward1
                                augment_obs2, augment_act2, augment_value2, augment_neglogp2, augment_done2, augment_reward2, _ = \
                                    self.rollout_subtask(next_state, goal, len(augment_obs_buf), goal)
                                if augment_obs2 is not None:
                                    print('Success')
                                    # augment_transition_buf += augment_transition2
                                    augment_obs_buf += augment_obs2
                                    augment_act_buf += augment_act2
                                    augment_value_buf += augment_value2
                                    augment_neglogp_buf += augment_neglogp2
                                    augment_done_buf += augment_done2
                                    augment_reward_buf += augment_reward2

                                    if augment_done_buf[0] != True:
                                        augment_done_buf[0] = True

                                    assert sum(augment_done_buf) == 1, augment_done_buf

                                    augment_returns = self.compute_adv(augment_value_buf, augment_done_buf, augment_reward_buf)
                                    assert augment_done_buf[0] == True
                                    assert sum(augment_done_buf) == 1
                                    # aug_obs, aug_act = zip(*augment_transition_buf)
                                    # print(len(augment_obs_buf), len(augment_act_buf), len(augment_neglogp_buf))
                                    # print(augment_adv_buf)
                                    # The augment data is directly passed to model
                                    # self.model.aug_obs += list(aug_obs)
                                    # self.model.aug_act += list(aug_act)
                                    for i in range(len(augment_obs_buf)):
                                        assert np.argmax(augment_obs_buf[i][-2:]) == 0
                                        assert np.argmax(augment_obs_buf[i][-7:-5]) == 0
                                    # self.model.aug_obs += augment_obs_buf
                                    # self.model.aug_act += augment_act_buf
                                    # self.model.aug_neglogp += augment_neglogp_buf
                                    # self.model.aug_adv += augment_adv_buf
                                    if self.model.aug_obs is None:
                                        self.model.aug_obs = np.array(augment_obs_buf)
                                        self.model.aug_act = np.array(augment_act_buf)
                                        self.model.aug_neglogp = np.array(augment_neglogp_buf)
                                        self.model.aug_value = np.array(augment_value_buf)
                                        self.model.aug_return = augment_returns
                                        self.model.aug_done = np.array(augment_done_buf)
                                    else:
                                        self.model.aug_obs = np.concatenate([self.model.aug_obs, np.array(augment_obs_buf)], axis=0)
                                        self.model.aug_act = np.concatenate([self.model.aug_act, np.array(augment_act_buf)], axis=0)
                                        self.model.aug_neglogp = np.concatenate([self.model.aug_neglogp, np.array(augment_neglogp_buf)],axis=0)
                                        self.model.aug_value = np.concatenate([self.model.aug_value, np.array(augment_value_buf)], axis=0)
                                        self.model.aug_return = np.concatenate([self.model.aug_return, augment_returns], axis=0)
                                        self.model.aug_done = np.concatenate([self.model.aug_done, np.array(augment_done_buf)], axis=0)
                                    assert self.model.aug_done[0] == True
                                # else:
                                #     print('Failed to achieve ultimate goal')
                            # else:
                            #     print('Failed to achieve subgoal')

                    # Then update buf
                    self.ep_state_buf[idx] = []
                    self.ep_transition_buf[idx] = []
            temp_time1 = time.time()
            augment_data_time += (temp_time1 - temp_time0)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))
        print('augment data takes', augment_data_time)
        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward

    def select_subgoal(self, transition_buf, k):
        # self.ep_transition_buf, self.model.value
        assert len(transition_buf) == 100, len(transition_buf)
        obs_buf, *_ = zip(*transition_buf)
        obs_buf = np.asarray(obs_buf)
        sample_t = np.random.randint(0, len(transition_buf), 4096)
        sample_obs = obs_buf[sample_t]
        noise = np.random.uniform(low=-0.15, high=0.15, size=(len(sample_t), 2))
        obstacle_xy = sample_obs[:, 6:8] + noise
        sample_obs[:, 6:8] = obstacle_xy
        sample_obs[:, 12:14] = sample_obs[:, 6:8] - sample_obs[:, 0:2]
        value2 = self.model.value(sample_obs)
        subgoal_obs = obs_buf[sample_t]
        subgoal_obs[:, 40:43] = subgoal_obs[:, 6:9]
        subgoal_obs[:, 43:45] = np.array([[0., 1.]])
        subgoal_obs[:, 45:47] = obstacle_xy
        subgoal_obs[:, 47:48] = subgoal_obs[:, 8:9]
        subgoal_obs[:, 48:50] = np.array([[0., 1.]])
        value1 = self.model.value(subgoal_obs)
        normalize_value1 = (value1 - np.min(value1)) / (np.max(value1) - np.min(value1))
        normalize_value2 = (value2 - np.min(value2)) / (np.max(value2) - np.min(value2))
        # best_idx = np.argmax(normalize_value1 * normalize_value2)
        ind = np.argsort(normalize_value1 * normalize_value2)
        good_ind = ind[-k:]
        # restart_step = sample_t[best_idx]
        # subgoal = subgoal_obs[best_idx, 45:50]
        restart_step = sample_t[good_ind]
        subgoal = subgoal_obs[good_ind, 45:50]
        # print('subgoal', subgoal, 'with value1', normalize_value1[best_idx], 'value2', normalize_value2[best_idx])
        # print('restart step', restart_step)
        return restart_step, subgoal

    def rollout_subtask(self, restart_state, goal, restart_step, ultimate_goal):
        aug_transition = []
        self.aug_env.unwrapped.sim.set_state(restart_state)
        self.aug_env.unwrapped.sim.forward()
        self.aug_env.unwrapped.goal[:] = goal
        dict_obs = self.aug_env.unwrapped.get_obs()
        obs = np.concatenate([dict_obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
        # print('subgoal', goal, 'obs', obs[-10:])
        def switch_goal(obs, goal):
            obs = obs.copy()
            assert len(goal) == 5
            obs[-5:] = goal
            goal_idx = np.argmax(goal[3:])
            obs[-10:-5] = np.concatenate([obs[3 + goal_idx * 3 : 6 + goal_idx * 3], goal[3:5]])
            return obs
        info = {'is_success': False}
        for step_idx in range(restart_step, 100):
            # If I use subgoal obs, value has problem
            # If I use ultimate goal obs, action should be rerunned
            # action, value, _, neglogpac = self.model.step(obs)
            action, _, _, _ = self.model.step(np.expand_dims(obs, axis=0))
            action = np.squeeze(action, axis=0)
            clipped_actions = action
            # Clip the actions to avoid out of bound error
            if isinstance(self.aug_env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(action, self.aug_env.action_space.low, self.aug_env.action_space.high)
            next_obs, _, _, info = self.aug_env.step(clipped_actions)
            self.model.num_aug_steps += 1
            reward = self.aug_env.compute_reward(switch_goal(next_obs, ultimate_goal), ultimate_goal, None)
            next_state = self.aug_env.unwrapped.sim.get_state()
            # aug_transition.append((obs, action, value, neglogpac, done, reward))
            aug_transition.append((switch_goal(obs, ultimate_goal), action, False, reward)) # Note that done refers to the output of previous action
            # print(step_idx, obs[-10:], next_obs[-10:])
            if info['is_success']:
                break
            obs = next_obs
        # print('length of augment transition', len(aug_transition))
        if info['is_success']:
            aug_obs, aug_act, aug_done, aug_reward = zip(*aug_transition)
            aug_obs = list(aug_obs)
            aug_act = list(aug_act)
            aug_done = list(aug_done)
            aug_reward = list(aug_reward)
            aug_neglogpac = self.model.sess.run(self.model.aug_neglogpac_op,
                                                {self.model.train_aug_model.obs_ph: np.array(aug_obs),
                                                 self.model.aug_action_ph: np.array(aug_act)})
            aug_value = self.model.value(np.array(aug_obs))
            # print(aug_neglogpac.shape)
            aug_neglogpac = aug_neglogpac.tolist()
            aug_value = aug_value.tolist()
            # print(np.mean(aug_neglogpac))
            return aug_obs, aug_act, aug_value, aug_neglogpac, aug_done, aug_reward, next_state
        return None, None, None, None, None, None, None

    def compute_adv(self, values, dones, rewards):
        if not isinstance(values, np.ndarray):
            values = np.asarray(values)
            dones = np.asarray(dones)
            rewards = np.asarray(rewards)
        # discount/bootstrap off value fn
        advs = np.zeros_like(rewards)
        last_gae_lam = 0
        for step in reversed(range(values.shape[0])):
            if step == values.shape[0] - 1:
                # Here we have assumed that the episode ends with done=Fase (not recorded in dones!).
                nextnonterminal = 0.0
                # So nextvalues here will not be used.
                nextvalues = np.zeros(values[0].shape)
            else:
                nextnonterminal = 1.0 - dones[step + 1]
                nextvalues = values[step + 1]
            delta = rewards[step] + self.gamma * nextvalues * nextnonterminal - values[step]
            advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        returns = advs + values
        return returns

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
