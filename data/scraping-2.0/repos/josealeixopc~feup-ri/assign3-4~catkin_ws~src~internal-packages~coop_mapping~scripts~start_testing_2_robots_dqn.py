#!/usr/bin/env python

from shutil import copyfile
import sys
import errno
import os
from datetime import datetime
import gym
import numpy
import time
from gym import wrappers

# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

# DQN
import dqn
import numpy as np

VS_ROS_DEBUG = 0
ENVS = ['TurtleBot3WorldMapping2RobotsTB3World-v0', 'TurtleBot3WorldMapping2RobotsHouse1-v0', 'TurtleBot3WorldMapping2RobotsHouse2-v0']
EPISODES = 10

ENV_VALUES = ['dev-no-gazebo', 'dev-gazebo', 'deploy']

def create_dir(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

def test(environment):
    """DQN agent implementation based from https://github.com/keon/deep-q-learning/blob/master/dqn.py
    """

    if VS_ROS_DEBUG:
        sys.stderr.write("Waiting for VS ROS debugger to be attached... Press a key and ENTER once it has been attached: ")
        raw_input()

    ### Export ENV variables BEGIN
    
    # Add node name to ROS logging messages 
    os.environ['ROSCONSOLE_FORMAT']='[${severity}] [${time}]: ${node}: ${message}'
    
    # Set TB3 model
    os.environ['TURTLEBOT3_MODEL'] = 'burger'

    # Set ENV variable
    if os.environ.get('ENV') is None:
        os.environ['ENV'] = ENV_VALUES[0]

    # Set testing variable
    os.environ['TEST'] = 'true'

    assert os.environ.get('ENV') in ENV_VALUES, "The ENV variable is not one of the allowable values: " + ','.join(ENV_VALUES)

    # Set WS path if no env variable is set
    if os.environ.get('ROS_WS') is None:
            os.environ['ROS_WS'] = '/home/jazz/Projects/FEUP/ProDEI/feup-ri/assign3-4/catkin_ws'
    
    ### Export ENV variables END

    rospy.init_node('turtlebot3_world_mapping_dqn', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = environment
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

    # Save starting time
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    
    # MAKE SURE TO USE loginfo INSTEAD OF logdebug! 
    # logdebug doesn't appear in \rosout for some reason (check rospy API), therefore it won't appear in rosconsole.
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('coop_mapping')

    results_dir = os.environ['ROS_WS']+ os.path.sep + 'tests' + os.path.sep + 'results' + os.path.sep + "{}-dqn".format(current_time)
    create_dir(results_dir)

    env = wrappers.Monitor(env, results_dir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    # Next, we build a very simple model.
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = dqn.DQNAgent(state_size, action_size)
    agent.load("/home/jazz/Documents/ri-results/tb3_world_400_episodes_85_percent/trainings/weights/2020-01-30-20-33-28-dqn_TurtleBot3WorldMapping2RobotsTB3World-v0_weights.h5f")
    done = False

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        rospy.logwarn("Initial state ==> {}".format(state))
        while True:
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            rospy.logwarn("Action taken ==> {}".format(action))

            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            rospy.logwarn("After taking action (s, r, d) ==> {}; {}; {}".format(state, reward, done))

            if done:
                break
    
        # Copy final map file to have a way of getting robot performance without being writing to file
        copyfile("/tmp/ros_merge_map.pgm", results_dir + os.path.sep + "final-map-episode-{}.pgm".format(e))

    env.close()
    
if __name__ == '__main__':
    test(ENVS[0])
    # test(ENVS[1])