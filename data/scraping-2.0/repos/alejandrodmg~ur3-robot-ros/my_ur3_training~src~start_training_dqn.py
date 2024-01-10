#!/usr/bin/env python3

import gym
import rospy
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from baselines import deepq

""" This implementation is a modfied version of OpenAI's cartpole example """

def callback(lcl, _glb):

    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def main():

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/ur3/task_and_robot_environment_name')
    # Create the Gym environment
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    rospy.logdebug("Gym environment done")

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
    print("Saving model to ur3_dqn.pkl")
    act.save("ur3_dqn.pkl")

if __name__ == '__main__':
    # Init node and run algorithm
    rospy.init_node('ur3_dqn', anonymous=True, log_level=rospy.WARN)
    main()
