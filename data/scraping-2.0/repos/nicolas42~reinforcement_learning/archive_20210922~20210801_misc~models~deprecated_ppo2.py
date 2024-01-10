'''
From OpenAI Baselines
'''
from models.base import Base
import os   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('DEBUG')
from scripts.distributions import make_pdtype
from scripts.mpi_running_mean_std import RunningMeanStd
from scripts.mpi_moments import mpi_moments
# from baselines.common import explained_variance, fmt_row, zipsame
import numpy as np
from gym import spaces
# import scripts.utils as U
import scripts.utils as TF_U
from scripts.mpi_adam_optimizer import MpiAdamOptimizer
from mpi4py import MPI
from scripts import logger
import time

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

class Policy():
    def __init__(self, ob, im, ob_space, ob_size, im_size, ac_space, sess, args, hid_size, normalize=True, const_std=False):
        self.ob = ob
        self.im = im
        self.ob_size = ob_size
        self.im_size = im_size
        self.sess = sess
        self.args = args
        self.pdtype = pdtype = make_pdtype(ac_space)
        self.const_std = const_std
        sequence_length = None
        if normalize:
            with tf.variable_scope("obfilter"):
                self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        if self.args.vis_type != "None":
            with tf.variable_scope("vis"):  
                x = tf.nn.relu(TF_U.conv2d(im, 16, "vis_l1", [8, 8], [4, 4], pad="VALID"))
                x = tf.nn.relu(TF_U.conv2d(x, 32, "vis_l2", [4, 4], [2, 2], pad="VALID"))
                x = TF_U.flattenallbut0(x)
                x = tf.nn.tanh(tf.layers.dense(x, 64, name='vis_lin', kernel_initializer=TF_U.normc_initializer(1.0)))
                    
        if normalize:
            obz = tf.clip_by_value((ob - tf.stop_gradient(self.ob_rms.mean)) / tf.stop_gradient(self.ob_rms.std), -5.0, 5.0)
        else:
            obz = ob
        
        if self.args.vis_type != "None":
            inputs = tf.concat(axis=1,values=[obz, x])
        else:
            inputs = obz

        with tf.variable_scope('vf'):
            
            if self.args.dual_value:
                last_out1 = tf.nn.tanh(tf.layers.dense(inputs, hid_size, name="fc1_1", kernel_initializer=TF_U.normc_initializer(1.0)))
                last_out1 = tf.nn.tanh(tf.layers.dense(last_out1, hid_size, name="fc2_1", kernel_initializer=TF_U.normc_initializer(1.0)))
                self.vpred1 = tf.layers.dense(last_out1, 1, name='final_1', kernel_initializer=TF_U.normc_initializer(1.0))[:,0]
                last_out2 = tf.nn.tanh(tf.layers.dense(inputs, hid_size, name="fc1_2", kernel_initializer=TF_U.normc_initializer(1.0)))
                last_out2 = tf.nn.tanh(tf.layers.dense(last_out2, hid_size, name="fc2_2", kernel_initializer=TF_U.normc_initializer(1.0)))
                self.vpred2 = tf.layers.dense(last_out2, 1, name='final_2', kernel_initializer=TF_U.normc_initializer(1.0))[:,0]
                self.vpred = tf.minimum(self.vpred1, self.vpred2)
            else:
                last_out = tf.nn.tanh(tf.layers.dense(inputs, hid_size, name="fc1", kernel_initializer=TF_U.normc_initializer(1.0)))
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc2", kernel_initializer=TF_U.normc_initializer(1.0)))
                self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=TF_U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            last_out = tf.nn.tanh(tf.layers.dense(inputs, hid_size, name='fc1', kernel_initializer=TF_U.normc_initializer(1.0)))
            last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc2', kernel_initializer=TF_U.normc_initializer(1.0)))
            self.mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=TF_U.normc_initializer(0.01))
            if self.const_std:
                logstd = tf.constant([np.log(0.75)]*int(pdtype.param_shape()[0]//2))
            else:      
                init = tf.constant_initializer([np.log(1.0)]*int(pdtype.param_shape()[0]//2))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2],initializer=init)

        pdparam = tf.concat([self.mean, self.mean * 0.0 + logstd], axis=1)
        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []
        self.stochastic = tf.placeholder(dtype=tf.bool, shape=())
        self.action = TF_U.switch(self.stochastic, self.pd.sample(), self.pd.mode())
        # self.action = self.pd.sample()
        self.neglogp = self.pd.neglogp(self.action)
        self.state = tf.constant([])
        self.initial_state = None

    def step(self, ob, im, stochastic=False):
        a, v, state, neglogp = self.sess.run([self.action, self.vpred, self.state, self.neglogp], 
                            feed_dict={self.ob: ob.reshape([1,self.ob_size]), 
                                        self.im: im.reshape([1]+self.im_size), 
                                        self.stochastic:stochastic})
        if state.size == 0:
            state = None   
        return a, v[0], state, neglogp[0]

class Model(Base):
    def __init__(self, name, env, ac_size, ob_size, im_size=[48,48,4], args=None, PATH=None, writer=None, hid_size=256, vis=False, normalize=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, mpi_rank_weight=1, max_timesteps=int(1e6), lr=3e-4, horizon=2048, batch_size=32, const_std=False):
        self.max_timesteps = max_timesteps
        self.horizon = horizon
        self.batch_size = batch_size
        self.learning_rate = lr
        self.env = env
        self.ac_size = ac_size

        self.sess = sess = TF_U.get_session()
        comm = MPI.COMM_WORLD
        self.name = name
        high = np.inf*np.ones(ac_size)
        low = -high
        ac_space = spaces.Box(low, high, dtype=np.float32)
        high = np.inf*np.ones(ob_size)
        low = -high
        ob_space = spaces.Box(low, high, dtype=np.float32)
        self.args = args   
        im_size = im_size
        self.PATH = PATH
        self.hid_size = hid_size
        self.writer = writer
        ob = TF_U.get_placeholder(name="ob" + name, dtype=tf.float32, shape=[None] + list(ob_space.shape))
        im = TF_U.get_placeholder(name="im" + name, dtype=tf.float32, shape=[None] + im_size)
        self.args = args
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            train_model = Policy(ob, im, ob_space, ob_size, im_size, ac_space, sess, args=args, normalize=normalize, hid_size=hid_size, const_std=const_std)
        Base.__init__(self)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Get the predicted value
        if self.args.dual_value:
            vf_loss1 = 0.5*tf.reduce_mean(tf.square(train_model.vpred1 - R))
            vf_loss2 = 0.5*tf.reduce_mean(tf.square(train_model.vpred2 - R))
            vf_loss = vf_loss1 + vf_loss2
        else:
            vf_loss = 0.5*tf.reduce_mean(tf.square(train_model.vpred - R))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        loss = pg_loss - entropy * ent_coef + vf_loss
        # vf_loss = vf_loss
        # UPDATE THE PARAMETERS USING LOSS
        
        params = tf.trainable_variables(name)
        # 2. Build our trainer
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)

        # 3. Calculate the gradients
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            grads_and_var = self.trainer.compute_gradients(loss, params)
            grads, var = zip(*grads_and_var)
        
        grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._train_op = self.trainer.apply_gradients(grads_and_var)
        
        std = tf.reduce_mean(train_model.pd.std)
        mean_ratio = tf.reduce_mean(ratio)
        mean_adv = tf.reduce_mean(self.ADV)
    
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 
        'clipfrac', 'std', 'ratio', 'adv', 'cliprange', 'learning_rate']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac, std, mean_ratio, mean_adv, CLIPRANGE, LR]

        self.train_model = train_model
        self.init_buffer()

    def train(self, epoch, lr, cliprange, obs, imgs, returns, masks, actions, values, neglogpacs,  exp_actions=None, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {
        self.train_model.ob : obs,
        self.A : actions,
        self.ADV : advs,
        self.R : returns,
        self.LR : lr,
        self.CLIPRANGE : cliprange,
        self.OLDNEGLOGPAC : neglogpacs,
        self.OLDVPRED : values
        }
        td_map[self.train_model.im] = imgs
    
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks
            
        return self.sess.run(self.stats_list + [self._train_op],td_map)[:-1]

    def run_train(self, data, last_value, last_done):
        self.finalise_buffer(data, last_value, last_done)
        self.train_model.ob_rms.update(self.data['ob'])
        if self.args.const_lr:
            self.cur_lrmult =  1.0
        else:
            self.cur_lrmult =  max(1.0 - float(self.timesteps_so_far) / self.max_timesteps, 0)
        lrnow = self.learning_rate*self.cur_lrmult
        self.lr = lrnow
        cliprangenow = 0.2
        self.n = self.data['done'].shape[0]
        inds = np.arange(self.n)
        t1 = time.time()
        for epoch in range(self.epochs):
            if self.enable_shuffle:
                np.random.shuffle(inds)
            self.loss = [] 
            for start in range(0, self.n, self.batch_size):
                end = start + self.batch_size
                mbinds = inds[start:end]
                slices = (self.data[key][mbinds] for key in self.training_input)
                outs = self.train(epoch, lrnow, cliprangenow, *slices)
                self.loss.append(outs[:len(self.loss_names)])
        self.loss = np.mean(self.loss, axis=0)
        self.evaluate(data['ep_rets'], data['ep_lens'])
        self.init_buffer()
        
    def step(self, ob, im, stochastic=False, multi=False): 
        if not multi:
            # import numpy as np
            # print(np.shape(im))
            actions, values, self.states, neglogpacs =  self.train_model.step(ob[None], im[None], stochastic=stochastic)
            return actions[0], values, self.states, neglogpacs
        else:
            actions, values, self.states, neglogpacs =  self.train_model.step(ob, im, stochastic=stochastic)
            return actions, values, self.states, neglogpacs        

    def get_advantage(self, last_value, last_done):    
        gamma = 0.99; lam = 0.95
        advs = np.zeros_like(self.data['rew'])
        lastgaelam = 0
        for t in reversed(range(len(self.data['rew']))):
            if t == len(self.data['rew']) - 1:
                nextnonterminal = 1.0 - last_done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - self.data['done'][t+1]
                nextvalues = self.data['value'][t+1]
            delta = self.data['rew'][t] + gamma * nextvalues * nextnonterminal - self.data['value'][t]
            advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        self.data['return'] = advs + self.data['value']
    
    def init_buffer(self):
        self.data_input = ['ob', 'im', 'ac', 'rew', 'done', 'value', 'neglogpac']
        self.training_input = ['ob', 'im', 'return', 'done', 'ac', 'value', 'neglogpac']
        self.data = {t:[] for t in self.data_input}

    def log_stuff(self, things):
        if self.rank == 0:
            for thing in things:
                self.writer.add_scalar(thing, things[thing], self.iters_so_far)
                logger.record_tabular(thing, things[thing])
        
    def add_to_buffer(self, data):
        ''' data needs to be a list of lists, same length as self.data'''
        for d,key in zip(data, self.data):
            self.data[key].append(d)

    def finalise_buffer(self, data, last_value, last_done):
        ''' data must be dict'''
        for key in self.data_input:
            if key == 'done':
                self.data[key] = np.asarray(self.data[key], dtype=np.bool)
            else:
                self.data[key] = np.asarray(self.data[key])
        self.n = next(iter(self.data.values())).shape[0]
        for key in data:
            self.data[key] = data[key]
        self.get_advantage(last_value, last_done)
        # if self.rank == 0:
        # self.log_stuff()
