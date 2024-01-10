#!/usr/bin/env python

import gym
import rospy
from baselines import deepq
from openai_ros.task_envs.turtlebot2 import turtlebot2_maze

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    rospy.init_node('turtlebot2_maze',anonymous=True,log_level=rospy.FATAL)
    env = gym.make("TurtleBot2Maze-v0")
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to turtlebot2.pkl")
    act.save("turtlebot2_model.pkl")


if __name__ == '__main__':
    main()
