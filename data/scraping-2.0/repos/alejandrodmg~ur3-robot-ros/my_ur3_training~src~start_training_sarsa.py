#!/usr/bin/env python3

import gym
import numpy
import time
import sarsa
from gym import wrappers
from functools import reduce
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment


if __name__ == '__main__':

    # Init node
    rospy.init_node('ur3_sarsa', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/ur3/task_and_robot_environment_name')
    # Create the Gym environment
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    rospy.logdebug("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_ur3_training')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/ur3/alpha")
    Epsilon = rospy.get_param("/ur3/epsilon")
    Gamma = rospy.get_param("/ur3/gamma")
    epsilon_discount = rospy.get_param("/ur3/epsilon_discount")
    nepisodes = rospy.get_param("/ur3/nepisodes")
    nsteps = rospy.get_param("/ur3/nsteps")

    # Initialises the algorithm that we are going to use for learning
    sarsa = sarsa.Sarsa(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = sarsa.epsilon
    start_time = time.time()
    highest_reward = 0
    rospy.logdebug("Starting to train the robot...")

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):

        cumulated_reward = 0
        done = False
        if sarsa.epsilon > 0.05:
            sarsa.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))

        # Initial action selection is outside the episode loop in SARSA
        action = sarsa.chooseAction(state)

        # For each episode, we test the robot for nsteps
        for i in range(nsteps):
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))
            # SARSA needs next_action, qlearn doesn't
            nextAction = sarsa.chooseAction(nextState)

            # Make the algorithm learn based on the results
            rospy.logwarn("# state we were=>" + str(state))
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            rospy.logwarn("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logwarn("# State in which we will start next step=>" + str(nextState))
            sarsa.learn(state, action, reward, nextState, nextAction)

            if not (done):
                state = nextState
                # Action is also updated here in SARSA
                action = nextAction
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.loginfo(("EP: " + str(x + 1) + " - [alpha: " + str(round(sarsa.alpha, 2)) + " - gamma: " + str(
            round(sarsa.gamma, 2)) + " - epsilon: " + str(round(sarsa.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(sarsa.alpha) + "|" + str(sarsa.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(
        reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
