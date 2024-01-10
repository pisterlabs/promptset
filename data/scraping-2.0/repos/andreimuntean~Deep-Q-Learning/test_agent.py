"""Tests a trained agent's ability to play Atari games from OpenAI Gym."""

import agent
import argparse
import environment
import logging
import random
import tensorflow as tf


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

PARSER = argparse.ArgumentParser()

PARSER.add_argument('env_name', help='name of the OpenAI Gym Atari environment that will be played')

PARSER.add_argument('load_path', help='loads the trained model from the specified path')

PARSER.add_argument('--save_path',
                    metavar='PATH',
                    help='path where to save experiments and videos',
                    default=None)

PARSER.add_argument('--render',
                    help='determines whether to display the game screen of the agent',
                    dest='render',
                    action='store_true',
                    default=False)

PARSER.add_argument('--num_episodes',
                    help='number of episodes to play',
                    type=int,
                    default=10)

PARSER.add_argument('--action_space',
                    nargs='+',
                    help='restricts the number of possible actions',
                    type=int)

PARSER.add_argument('--max_episode_length',
                    metavar='TIME STEPS',
                    help='maximum number of time steps per episode',
                    type=int,
                    default=50000)

PARSER.add_argument('--epsilon',
                    metavar='EPSILON',
                    help='likelihood that the agent selects a random action',
                    type=float,
                    default=0.0)

PARSER.add_argument('--observations_per_state',
                    metavar='FRAMES',
                    help='number of consecutive frames within a state',
                    type=int,
                    default=3)

PARSER.add_argument('--gpu_memory_alloc',
                    metavar='PERCENTAGE',
                    help='determines how much GPU memory to allocate for the neural network',
                    type=float,
                    default=0.1)


def main(args):
    """Loads a trained agent that plays Atari games from OpenAI Gym."""

    env = environment.AtariWrapper(args.env_name,
                                   args.max_episode_length,
                                   100,
                                   args.observations_per_state,
                                   args.action_space,
                                   args.save_path)

    with tf.Session() as sess:
        player = agent.TestOnlyAgent(env)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, args.load_path)

        for _ in range(args.num_episodes):
            episode_reward = 0

            for t in range(args.max_episode_length):
                # Occasionally try a random action.
                if random.random() < args.epsilon:
                    action = env.sample_action()
                else:
                    action = player.get_action(env.get_state())

                reward = env.step(action)
                episode_reward += reward

                if args.render:
                    env.render()

                if env.done:
                    break

            LOGGER.info('Episode finished after %d time steps. Reward: %d.', t + 1, episode_reward)


if __name__ == '__main__':
    main(PARSER.parse_args())
