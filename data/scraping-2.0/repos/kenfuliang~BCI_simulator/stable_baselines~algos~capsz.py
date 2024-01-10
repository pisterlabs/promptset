import time
from contextlib import contextmanager
from collections import deque


import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter, fmt_row, dataset
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.schedules import get_schedule_fn, constfn
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.ppo2.ppo2 import PPO2, swap_and_flatten, safe_mean

from stable_baselines.common.distributions import DiagGaussianProbabilityDistribution


class PPOCAPSZ(PPO2):
    '''
    modify from PPO2. In this algorithm, we implemented CAPSZ(conditional on action policy smoothness and zero).  
    '''
    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, d_stepsize = 3e-4, vf_coef=0.5,smooth_coef=0, zero_coef=0, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, random_initial=False):
        self.zero_coef = zero_coef
        self.smooth_coef = smooth_coef
        super().__init__(policy=policy, env=env, gamma=gamma, n_steps=n_steps, ent_coef=ent_coef, learning_rate = learning_rate, d_stepsize=d_stepsize, vf_coef=vf_coef, max_grad_norm=max_grad_norm, lam=lam, nminibatches=nminibatches, noptepochs=noptepochs, cliprange_vf=cliprange_vf, verbose=verbose,  _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)



    def _make_runner(self):
        return Runner(env=self.env, model=self, n_steps=self.n_steps,
                      gamma=self.gamma, lam=self.lam,
                      reward_giver=self.reward_giver,
                      expert_dataset=self.expert_dataset,
                      random_initial=self.random_initial)

    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action

    def setup_model(self):
        #from stable_baselines.gail.adversary import TransitionClassifier
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_steps

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)
                
                if self.using_gail:
                    self.reward_giver = None#TransitionClassifier(self.observation_space, self.action_space,
                                            #                 self.hidden_size_adversary,
                                            #                 entcoeff=self.adversary_entcoeff,reward_setting=self.reward_setting)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
                        "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, reuse=False, **self.policy_kwargs)

                with tf.variable_scope("train_model", reuse=True,custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)

                with tf.variable_scope("train_model", reuse=True,custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model_next_obs = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)



                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float64, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float64, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float64, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float64, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float64, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float64, [], name="clip_range_ph")

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
                        self.clip_range_vf_ph = tf.placeholder(tf.float64, [], name="clip_range_vf_ph")

                    if self.clip_range_vf_ph is None:
                        # No clipping
                        vpred_clipped = train_model.value_flat
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        vpred_clipped = self.old_vpred_ph + \
                            tf.clip_by_value(train_model.value_flat - self.old_vpred_ph,
                                             - self.clip_range_vf_ph, self.clip_range_vf_ph)


                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)

                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpred_clipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 + self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),self.clip_range_ph), tf.float64))

                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                    ## CAPS
                    self.temporal_loss = train_model.proba_distribution.kl(train_model_next_obs.proba_distribution)

                    ## CAPZ
                    zero_gaussian = DiagGaussianProbabilityDistribution(np.zeros(4))
                    self.zero_loss = train_model.proba_distribution.kl(zero_gaussian)

                    loss_CAPS = self.smooth_coef * self.temporal_loss
                    loss_CAPZ = self.zero_coef * self.zero_loss

                    loss = loss + loss_CAPZ + loss_CAPS

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
                    g_grads = list(zip(grads, self.params))


                if self.using_gail:
                    if self.reward_setting=='GAN':
                        g_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                        d_trainer = tf.train.AdamOptimizer(learning_rate=self.d_stepsize, epsilon=1e-5)
                    elif (self.reward_setting=='WGAN' or self.reward_setting=='WGAN-GP'):
                        g_trainer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate_ph)
                        d_trainer = tf.train.RMSPropOptimizer(learning_rate=self.d_stepsize)
                    d_grads = self.reward_giver.get_gradient()
                    self._d_train = d_trainer.apply_gradients(d_grads)
                    self._g_train = g_trainer.apply_gradients(g_grads)
                else:
                    g_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                    self._g_train = g_trainer.apply_gradients(g_grads)




                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    if self.clip_range_vf_ph is not None:
                        tf.summary.scalar('clip_range_vf', tf.reduce_mean(self.clip_range_vf_ph))

                    tf.summary.scalar('old_neglog_action_probability', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probability', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)


                if self.using_gail:
                    self.params.extend(self.reward_giver.get_trainable_variables())

                self.train_model = train_model ## target policy which is going to be updated.
                self.train_model_next_obs = train_model_next_obs
                self.act_model = act_model ## behavior policy which used to interact with env
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary_input_info = tf.summary.merge_all(scope='input_info')
                self.summary_loss = tf.summary.merge_all(scope='loss')
                self.summary = tf.summary.merge([self.summary_input_info,self.summary_loss])

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, obs_next, update,
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
        #print(returns.shape,values.shape)
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        ## feed_dict for tf.sess
        #print(actions.shape)
        td_map = {self.train_model.obs_ph: obs, 
                  self.train_model_next_obs.obs_ph: obs_next, 
                  self.action_ph: actions,
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
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._g_train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._g_train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            ## Calculate losses, entropy, kl, clipfrac, and apply gradient to train model parameters
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._g_train], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def _train_d_step(self, ob_batch, ac_batch, ob_expert, ac_expert,update,writer):
        td_map = {self.reward_giver.generator_obs_ph: ob_batch, 
                  self.reward_giver.generator_acs_ph: ac_batch,
                  self.reward_giver.expert_obs_ph: ob_expert,
                  self.reward_giver.expert_acs_ph: ac_expert}
        *d_loss,summary = self.sess.run(self.reward_giver.losses+[self.reward_giver.summary_d_loss_op],td_map)
        expert_acc = d_loss[5]
        generator_acc = d_loss[4]
        writer.add_summary(summary,update)
        self.sess.run([self._d_train],td_map)

        return d_loss
        

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="PPOCAPSZ",
              reset_num_timesteps=True):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            with self.sess.as_default():

                if self.using_gail:
                    true_reward_buffer = deque(maxlen=40)
                    
                    # Initialize dataloader
                    #batchsize = self.n_batch#self.timesteps_per_batch // self.d_step
                    self.expert_dataset.init_dataloader(self.n_batch)
                    


                t_first_start = time.time()

                n_updates = total_timesteps // self.n_batch
                for update in range(1, n_updates + 1):

                    assert self.n_batch % self.nminibatches == 0, ("The number of minibatches (`nminibatches`) "
                                                                   "is not a factor of the total number of samples "
                                                                   "collected per rollout (`n_batch`), "
                                                                   "some samples won't be used."
                                                                   )

                    
                    ## -- optimizing generator -- ##
                    logger.log("Optimizing Generator...")
                    batch_size = self.n_batch // self.nminibatches
                    t_start = time.time()
                    frac = 1.0 - (update - 1.0) / n_updates
                    lr_now = self.learning_rate(frac)
                    cliprange_now = self.cliprange(frac)
                    cliprange_vf_now = cliprange_vf(frac)

                    obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward, obs_next = self.runner.run()
                    self.num_timesteps += self.n_batch
                    self.ep_info_buf.extend(ep_infos)
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
                                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs, obs_next))
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
                                slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs, obs_next))
                                mb_states = states[mb_env_inds]
                                mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, update=timestep,
                                                                     writer=writer, states=mb_states,
                                                                     cliprange_vf=cliprange_vf_now))

                    loss_vals = np.mean(mb_loss_vals, axis=0)

                    ## -- optimizing discriminator -- ##
                    #if self.using_gail:
                    #    logger.log("Optimizing Discriminator...")
                    #    logger.log(fmt_row(13, self.reward_giver.loss_name))
                    #    batch_size = len(obs)#//self.d_step#1024#self.timesteps_per_batch // self.d_step
                    #    d_losses = []
                    #    for _ in range(self.d_step):
                    #        for ob_batch, ac_batch in dataset.iterbatches((obs, actions),
                    #                                                       include_final_partial_batch=False,
                    #                                                       batch_size=batch_size,
                    #                                                       shuffle=True):
                    #            ob_expert, ac_expert = self.expert_dataset.get_next_batch()
                    #            # update running mean/std for reward_giver
                    #            if self.reward_giver.normalize:
                    #                self.reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))

                    #            # Reshape actions if needed when using discrete actions
                    #            if isinstance(self.action_space, gym.spaces.Discrete):
                    #                if len(ac_batch.shape) == 2:
                    #                    ac_batch = ac_batch[:, 0]
                    #                if len(ac_expert.shape) == 2:
                    #                    ac_expert = ac_expert[:, 0]
                    #            d_losses.append(self._train_d_step(ob_batch,ac_batch,ob_expert,ac_expert,update=self.num_timesteps,writer=writer))
                    #        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

                   
                    
                    t_now = time.time()
                    fps = int(self.n_batch / (t_now - t_start))

                    #print(self.episode_reward)
                    if writer is not None:
                        total_episode_reward_logger(self.episode_reward,
                                                    true_reward.reshape((self.n_envs, self.n_steps)),
                                                    masks.reshape((self.n_envs, self.n_steps)),
                                                    writer, self.num_timesteps)

                    #print(self.episode_reward)
                    if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                        explained_var = explained_variance(values, returns)
                        logger.logkv("serial_timesteps", update * self.n_steps)
                        logger.logkv("n_updates", update)
                        logger.logkv("total_timesteps", self.num_timesteps)
                        logger.logkv("fps", fps)
                        logger.logkv("explained_variance", float(explained_var))
                        if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                            logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                            logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                        logger.logkv('time_elapsed', t_start - t_first_start)
                        for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                            logger.logkv(loss_name, loss_val)
                        logger.dumpkvs()

                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break

            return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "zero_coef": self.zero_coef,
            "smooth_coef": self.smooth_coef,


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
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)




class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam, reward_giver, expert_dataset, **kwargs):
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
        self.reward_giver = reward_giver
        self.expert_dataset = expert_dataset
        self.random_initial = kwargs.get('random_initial')

        self.summary()
    def summary(self):
        if self.random_initial is not None:
            print("random_initial:",self.random_initial)



    def _run(self):
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
        mb_obs,mb_obs_next, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], [], []
        mb_true_rewards = []
        mb_states = self.states
        ep_infos = []
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            assert actions.shape[0]==self.model.n_envs
            
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            if(self.reward_giver!=None):
                rewards = self.reward_giver.get_reward(self.obs, clipped_actions)
                self.obs[:], true_rewards, self.dones, infos = self.env.step(clipped_actions)
                assert len(np.where(true_rewards<0)[0])==0

                if(self.random_initial):
                    for env_id, done in enumerate(self.dones):
                        if done:
                            starting_idx    =np.random.randint(0,self.expert_dataset.num_transition)
                            self.env.stackedobs[env_id]  =self.expert_dataset.observations[starting_idx]
                            cursor_p        =self.expert_dataset.observations[starting_idx][0:2,-1]
                            cursor_v        =self.expert_dataset.observations[starting_idx][2:4,-1]
                            target          =self.expert_dataset.observations[starting_idx][4:6,-1]
                            self.env.set_attr('cursor_position',cursor_p,env_id)
                            self.env.set_attr('cursor_velocity',cursor_v,env_id)
                            self.env.set_attr('hand_position',cursor_p,env_id)
                            self.env.set_attr('hand_velocity',cursor_v,env_id)
                            self.env.set_attr('target_position',target,env_id)
                            self.env.set_attr('task_state',2,env_id)

            else:
                self.obs[:], true_rewards, self.dones, infos = self.env.step(clipped_actions)
                rewards = true_rewards

            mb_obs_next.append(self.obs.copy())
            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
            mb_true_rewards.append(true_rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_obs_next = np.asarray(mb_obs_next, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float64)
        mb_true_rewards = np.asarray(mb_true_rewards, dtype=np.float64)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float64)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float64)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)

        if((mb_rewards.ndim)==1):
            mb_rewards = np.expand_dims(mb_rewards,1)
        assert mb_rewards.shape==mb_true_rewards.shape, "mb_rewards.shape = {} and mb_true_rewards.shape = {}".format(mb_rewards.shape,mb_true_rewards.shape)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_true_rewards)
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

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward, mb_obs_next = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward, mb_obs_next))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward, mb_obs_next


