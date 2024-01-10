#!/usr/bin/env python3
import argparse
import gym
import numpy as np
import tensorflow as tf
from itertools import count
from collections import namedtuple

parser = argparse.ArgumentParser(description='TensorFlow REINFORCE')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 100)')
parser.add_argument('--render_interval', type=int, default=-1, metavar='N',
                    help='interval between rendering (default: -1)')
parser.add_argument('--env_id', type=str, default='LunarLander-v2',
                    help='gym environment to load')
args = parser.parse_args()

"""

This file implements the standard vanilla REINFORCE algorithm, also
known as Monte Carlo Policy Gradient. 

This copies from the OpenAI baselines structure, which I found to be a bit
confusing at first, but actually quite nice and clean. (Tensorflow is just a
huge pain to learn, but once you do, it is not as bad.)


    Resources:
        Sutton and Barto: http://incompleteideas.net/book/the-book-2nd.html
        Karpathy blog: http://karpathy.github.io/2016/05/31/rl/
        OpenAI baselines PPO algorithm: https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py


    Glossary:
        (logits) = numerical policy preferences, or unnormalized probailities of actions
                        or last layer neural net
"""




# HELPERS
def calculate_discounted_returns(rewards):
    """
    Calculate discounted reward and then normalize it
    (see Sutton book for definition)
    Params:
        rewards: list of rewards for every episode
    """
    returns = np.zeros(len(rewards))

    next_return = 0 # 0 because we start at the last timestep
    for t in reversed(range(0, len(rewards))):
        next_return = rewards[t] + args.gamma * next_return
        returns[t] = next_return
    # normalize for better statistical properties
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps)
    return returns

def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer

# Class organization copied (roughly) from OpenAI baselines
class PolicyNetwork(object):
    def __init__(self, ob_n, ac_n, hidden_dim=200, name='policy_network'):
        with tf.variable_scope(name):
            self._init(ob_n, ac_n, hidden_dim)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_n, ac_n, hidden_dim):
        self.ob_n = ob_n
        self.ac_n = ac_n

        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, ob_n])

        x = tf.layers.dense(inputs=self.obs, units=hidden_dim, activation=tf.nn.relu, name='hidden')
        self.logits = tf.layers.dense(inputs=x, units=self.ac_n, activation=None, kernel_initializer=normc_initializer(0.01), name='logits')

        ac = self._sample()
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        self._act = self._run_gen(self.obs, ac)

    def act(self, ob):
        ac1 = self._act(ob[None])
        return ac1

    def _sample(self):
        """Random sample an action"""
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    def _run_gen(self, ob, ac):
        def run(ob_feed):
            """Run an observation through the nn to get an action
            this will only be used to run the policy.  To train it, we
            later feed in the observations, selected actions, and rewards all at once"""
            results = tf.get_default_session().run(ac, feed_dict={ob:ob_feed})
            return results
        return run

    def neglogp(self, x):
        """This computes the negative log probability of the given action.
        It is used to pass the gradient back through the network for training
        (in tf speak, this is the loss that we minimize)

        NOTE: when we evaluate this, we are refeeding all of the observations,
        chosen actions, and rewards back through the network.  Meaning we don't worry
        about caching when we are running the env. This is just for ease in tensorflow.
        """
        one_hot_actions = tf.one_hot(x, self.ac_n)
        # see http://cs231n.github.io/linear-classify/#softmax
        # and http://karpathy.github.io/2016/05/31/rl/
        # The math matches up because we are using the softmax to sample actions

        # why is the chosen action the label?
        # the chosen action is the label because that would create the signal that always
        # make that one more probable. since we multiply this by the return signal, that will
        # good actions more probable and bad actions less probable.
        return tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, 
                labels=one_hot_actions)


class REINFORCE(object):
    """
    Object to handle running the algorithm. Uses a PolicyNetwork
    """
    def __init__(self, env):
        self.ob_n = env.observation_space.shape[0]
        self.ac_n = env.action_space.n

        self.pi = PolicyNetwork(self.ob_n, self.ac_n)

        self.obs = self.pi.obs
        self.ac = tf.placeholder(tf.int32, shape=[None], name='ac')
        self.atarg = tf.placeholder(tf.float32, shape=[None], name='atarg')
        self.loss = self.atarg * self.pi.neglogp(self.ac)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        # TODO: was updating the training pipeline to match baselines
        # TODO: i may just want to copy baselines and add in baby algorithms. fork it and call
        # it baby baselines. REINFORCE, AC, and commented like shit. A ramp up to baselines 
        # proper

    def select_action(self, obs, sess=None):
        """
        Run observation through network and sample an action to take. Keep track
        of dh to use to update weights
        """
        sess = sess or tf.get_default_session()
        return self.pi.act(obs)
    
    def update(self, ep_cache, sess=None):
        returns = calculate_discounted_returns(ep_cache.rewards)
        obs = np.array(ep_cache.obs)
        taken_actions = np.array(ep_cache.actions)

        sess = sess or tf.get_default_session()
        feed_dict = {self.obs: obs, self.ac: taken_actions, self.atarg: returns}
        sess.run([self.train_op], feed_dict=feed_dict)

def main():
    """Run REINFORCE algorithm to train on the environment"""

    EpCache = namedtuple("EpCache", ["obs", "actions", "rewards"])
    avg_reward = []
    for i_episode in count(1):
        ep_cache = EpCache([], [], [])
        obs = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = reinforce.select_action(obs)[0]

            ep_cache.obs.append(obs)
            ep_cache.actions.append(action)

            obs, reward, done, _ = env.step(action)
            
            ep_cache.rewards.append(reward)

            if args.render_interval != -1 and i_episode % args.render_interval == 0:
                env.render()

            if done:
                break

        reinforce.update(ep_cache)

        if i_episode % args.log_interval == 0:
            print("Ave reward: {}".format(sum(avg_reward)/len(avg_reward)))
            avg_reward = []
        else:
            avg_reward.append(sum(ep_cache.rewards))

if __name__ == '__main__':
    env = gym.make(args.env_id)
    env.seed(args.seed)
    np.random.seed(args.seed)
    reinforce = REINFORCE(env)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        main()

