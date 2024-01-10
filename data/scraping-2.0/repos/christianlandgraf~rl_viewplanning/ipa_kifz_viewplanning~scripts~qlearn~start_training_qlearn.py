#!/usr/bin/env python
'''
The Q Learning Training Process is based on
https://bitbucket.org/theconstructcore/openai_examples_projects/src/master/turtle2_openai_ros_example/scripts/start_qlearning.py

'''

import os.path
import sys
import gym
import numpy as np
import pandas as pd
import csv
import time
import datetime
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

import torch
from tensorboardX import SummaryWriter

if __name__ == '__main__':

    rospy.init_node('ipa_kifz_viewplanning_qlearn',
                    anonymous=True,
                    log_level=rospy.WARN)

    # Init OpenAI_ROS Environment
    rospy.loginfo("Init Task Environment")
    task_and_robot_environment_name = rospy.get_param(
        '/ipa_kifz/task_and_robot_environment_name')
    rospy.loginfo("Init ROS Environment")
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ipa_kifz_viewplanning')

    from helper import ipa_kifz_logging
    logfile_time = datetime.datetime.now()
    qLearnLogger = ipa_kifz_logging.Logger(logfile_time, log_freq=2)
    qLearnLogger.save_params()
    env = wrappers.Monitor(env, qLearnLogger.log_directory, force=True)
    rospy.loginfo("Monitor Wrapper started")
    writer = SummaryWriter(qLearnLogger.log_directory)

    last_time_steps = np.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    alpha = rospy.get_param("/ipa_kifz/alpha")
    epsilon = rospy.get_param("/ipa_kifz/epsilon")
    gamma = rospy.get_param("/ipa_kifz/gamma")
    epsilon_discount = rospy.get_param("/ipa_kifz/epsilon_discount")
    nepisodes = rospy.get_param("/ipa_kifz/nepisodes")
    max_nsteps = rospy.get_param("/ipa_kifz/max_nsteps")

    # running_step = rospy.get_param("/ipa_kifz/running_step")

    # Initialises the algorithm that we are going to use for learning
    rospy.logdebug("############### ACTION SPACE =>" + str(env.action_space))

    # init Q-learning environment
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           epsilon=epsilon,
                           alpha=alpha,
                           gamma=gamma)
    initial_epsilon = qlearn.epsilon

    # Discrete Action Space!
    states = []
    for action in range(env.action_space.n):
        state = ' '.join(
            map(str, [
                env.discretized_action_space[action][0][0],
                env.discretized_action_space[action][0][1],
                env.discretized_action_space[action][0][2],
                np.around(env.discretized_action_space[action][0][3], 2),
                np.around(env.discretized_action_space[action][0][4], 2),
                np.around(env.discretized_action_space[action][0][5], 2),
                np.around(env.discretized_action_space[action][0][6], 2)
            ]))
        states.append(state)
    states.append(' '.join(map(
        str, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])))  # initial state
    qlearn.initQ(states, range(env.action_space.n))

    start_time = time.time()
    highest_reward = 0
    reward_list = []

    # Initialize Logger

    # Start training loop
    for x in range(nepisodes):
        rospy.logdebug("############### START EPISODE=>" + str(x))

        # Episode Init: Reset the environment and get first state of the robot
        observation = env.reset()
        cumulated_reward = 0
        done = False
        observation[observation == 0.] = 0.0  #Normalize -0. to 0.
        state_0 = ' '.join(
            map(str, [
                observation[0], observation[1], observation[2], observation[3],
                observation[4], observation[5], observation[6]
            ]))
        # state_0 = env.init_state(discretized_actions)

        # decrease epsilon in each episode
        if qlearn.epsilon > 0.01:
            qlearn.epsilon *= epsilon_discount

        # Show the actual robot pose on screen
        # env.render()
        # each episode, the robot does not more than max_nsteps with measurements
        previous_actions = []
        episode_poses = []

        for i in range(max_nsteps):
            rospy.logwarn("############### Start Step=>" + str(i))

            # Pick an action based on the current state
            action, previous_actions = qlearn.chooseAction(
                state_0, previous_actions)
            rospy.logwarn("Next action is:" + str(action))

            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)
            state_1 = ' '.join(
                map(str, [
                    observation[0], observation[1], observation[2],
                    observation[3], observation[4], observation[5],
                    observation[6]
                ]))

            # cumulate reward
            if reward < 0:
                reward = 0
            cumulated_reward += reward

            writer.add_scalar('episode_reward', cumulated_reward, x)

            # Make the algorithm learn based on the results
            rospy.logwarn("# state we were=>" + str(state_0))
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            rospy.logwarn("# episode cumulated_reward=>" +
                          str(cumulated_reward))
            rospy.logwarn("# Next state=>" + str(state_1))
            qlearn.learn(state_0, action, reward, state_1)

            # Save poses for logging
            episode_poses.append(observation[:-1])

            # Check if done
            if not (done):
                rospy.logwarn("NOT DONE")
                state_0 = state_1
            else:
                rospy.logwarn("DONE")
                last_time_steps = np.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step=>" + str(i))

        # rospy.logwarn("# updated q-table after episode " + str(x) + "=>" + str(qlearn.q))
        reward_list.append(cumulated_reward)

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logwarn(
            ("EP: " + str(x + 1) + " - [alpha: " +
             str(round(qlearn.alpha, 2)) + " - gamma: " +
             str(round(qlearn.gamma, 2)) + " - epsilon: " +
             str(round(qlearn.epsilon, 2)) + "] - Reward: " +
             str(cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

        qLearnLogger.log_episode(cumulated_reward, episode_poses, qlearn.q)

    rospy.loginfo(
        ("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" +
         str(qlearn.gamma) + "|" + str(initial_epsilon) + "*" +
         str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    qLearnLogger.save_model(qlearn.q)

    # # print("Parameters: a="+str)
    # rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    # rospy.loginfo("Best 100 score: {:0.2f}".format(
    #     reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    writer.close()
    env.close()
