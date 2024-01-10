#!/usr/bin/env python

import gym
import numpy
import time
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from functools import reduce
import pickle


if __name__ == '__main__':

    rospy.init_node('example_turtlebot2_maze_qlearn', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot2/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('turtle2_openai_ros_example')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/turtlebot2/alpha")
    Epsilon = rospy.get_param("/turtlebot2/epsilon")
    Gamma = rospy.get_param("/turtlebot2/gamma")
    epsilon_discount = rospy.get_param("/turtlebot2/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot2/nepisodes")
    nsteps = rospy.get_param("/turtlebot2/nsteps")

    running_step = rospy.get_param("/turtlebot2/running_step")
    rospy.logwarn("env.action_space %s" % env.action_space)
    rospy.logwarn("env.action_space.n %s" % env.action_space.n)
    rospy.logwarn("range(env.action_space.n) %s" % range(env.action_space.n))
    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0
    q_file_mid = open("/home/eldar/python3_ws/src/turtle2_openai_ros_example/src/qdictmid.pkl", "wb")
    q_file_end = open("/home/eldar/python3_ws/src/turtle2_openai_ros_example/src/qdict.pkl", "wb")
    #env._max_episode_steps = nsteps

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes-1):
        rospy.logdebug("############### START EPISODE=>" + str(x))

        cumulated_reward = 0
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        #for i in range(nsteps-1):
        i = 0
        while (i < nsteps) and (not done):
            rospy.logwarn("############### Start Step=>" + str(i) + " of episode ==> " + str(x))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            rospy.logwarn("Next action is:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            rospy.logwarn(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            rospy.logwarn("# state we were=>" + str(state))
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            rospy.logwarn("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logwarn("# State in which we will start next step=>" + str(nextState))
            qlearn.learn(state, action, reward, nextState)

            if not (done):
                rospy.logwarn("NOT DONE")
                state = nextState
            else:
                rospy.logwarn("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                #break
            rospy.logwarn("############### END Step=>" + str(i))
            i += 1
           
            #raw_input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)
        if x == (nepisodes - 2):
            q_dict = qlearn.returnQ()
            pickle.dump(q_dict, q_file_end)
            q_file_end.close()
            rospy.logwarn("Saving final pickle")
        elif x == 200:
            q_dict = qlearn.returnQ()
            pickle.dump(q_dict, q_file_mid)
            q_file_mid.close() 
            rospy.logwarn("Saving pickle for 200 episodes")   
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo((" nepisodes " + str(nepisodes) + " alpha " + str(qlearn.alpha) + " gamma " + str(qlearn.gamma) + " initial epsilon " + str(
        initial_epsilon) + " epsilon discount " + str(epsilon_discount) + " highest reward " + str(highest_reward) + " "))

    l = last_time_steps.tolist()
    l.sort()
 


    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
