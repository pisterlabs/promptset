import numpy as np
import tensorflow as tf
import gym

from atari_wrappers import wrap_deepmind
from ppo import PPO, DEFAULT_LOGDIR


def ortho_init(scale=1.0):
    # (copied from OpenAI baselines)
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


class AtariPPO(PPO):
    input_shape = [84,84,2]
    input_type = np.uint8
    video_freq = 100
    num_env = 4
    gamma = 0.99
    lmda = 0.95
    learning_rate = 3e-4
    entropy_reg = 0.0
    vf_coef = 1.0
    max_gradient_norm = 1.0
    eps_clip = 0.1
    reward_clip = 1.0
    end_of_life_penalty = 0.0

    def __init__(self, env_name, params={}, **kwargs):
        self.load_params(params)
        envs = [gym.make(env_name) for _ in range(self.num_env)]
        envs[0] = gym.wrappers.Monitor(
            envs[0], kwargs.get('logdir', DEFAULT_LOGDIR),
            force=True, video_callable=lambda t: t % self.video_freq == 0)
        num_frames = self.input_shape[-1]
        envs = [
            wrap_deepmind(env, num_frames, self.end_of_life_penalty)
            for env in envs
        ]
        super().__init__(envs, **kwargs)

    def build_logits_and_values(self, img_in):
        y = tf.cast(img_in, tf.float32) / 255.0
        y = tf.layers.conv2d(
            y, filters=32, kernel_size=8, strides=4,
            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
        )
        self.op.layer1 = y
        y = tf.layers.conv2d(
            y, filters=64, kernel_size=4, strides=2,
            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
        )
        self.op.layer2 = y
        y = tf.layers.conv2d(
            y, filters=64, kernel_size=3, strides=1,
            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
        )
        self.op.layer3 = y
        y = tf.layers.dense(
            tf.layers.flatten(y), units=512,
            activation=tf.nn.relu, kernel_initializer=ortho_init(np.sqrt(2)),
        )
        self.op.layer4 = y
        logits = tf.layers.dense(
            y, units=self.envs[0].action_space.n,
            kernel_initializer=ortho_init(0.01))
        values = tf.layers.dense(
            y, units=1,
            kernel_initializer=ortho_init(1.0))[:,0]

        def dead_fraction(x):
            x = tf.equal(x, 0.0)
            x = tf.cast(x, tf.float32)
            return tf.reduce_mean(x)

        with tf.name_scope('is_dead'):
            tf.summary.scalar('layer1', dead_fraction(self.op.layer1))
            tf.summary.scalar('layer2', dead_fraction(self.op.layer2))
            tf.summary.scalar('layer3', dead_fraction(self.op.layer3))
            tf.summary.scalar('layer4', dead_fraction(self.op.layer4))

        return logits, values


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default=DEFAULT_LOGDIR)
    parser.add_argument('--env')
    parser.add_argument('--pfile')
    parser.add_argument('--load')
    args = parser.parse_args()
    if args.pfile:
        with open(args.pfile, 'r') as f:
            params = json.load(f)
    else:
        params = {}
    if args.env:
        params['env'] = ''.join(x.capitalize() for x in args.env.split('-'))
        params['env'] += 'Deterministic-v4'
    model = AtariPPO(
        params['env'],
        logdir=args.logdir,
        saver_args=params.get('saver_args', {}),
        params=params.get('params', {}))
    if args.load:
        model.saver.restore(model.session, args.load)
    model.train(**params.get('trainer_args', {}))
