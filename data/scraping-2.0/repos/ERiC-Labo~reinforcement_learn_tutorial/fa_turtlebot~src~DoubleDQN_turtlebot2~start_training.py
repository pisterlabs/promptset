#!/usr/bin/env python3
#Import Libraries
import gym
from gym import wrappers
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from d_dqn import *

if __name__ == '__main__':

	#Initialize ROS node
    rospy.init_node('example_turtlebot2_maze_qlearn',
                    anonymous=True, log_level=rospy.WARN)
    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot2/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('fa_turtlebot')
    outdir = pkg_path + '/training_results'
    print(outdir)
    env = wrappers.Monitor(env, outdir, force=True)

    #env_id = "CartPole-v0"
    
    MAX_EPISODES = rospy.get_param("~max_episode", 30)
    MAX_STEPS = rospy.get_param("~max_step", 500)
    BATCH_SIZE = rospy.get_param("~batch_size", 64)

    agent = DQNAgent_d_dqn(env, use_conv=False)
    episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, 
        MAX_STEPS, BATCH_SIZE)
    env.close()