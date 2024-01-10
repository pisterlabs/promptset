#!/usr/bin/env python
import gym
from gym import wrappers
import numpy
# from openai_examples_projects.my_turtlebot3_openai_example.scripts.dqn import DQNSolver
from dqn import DQNSolver
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import signal
import os

# [Tri Huynh] Use saved trained q_table in q_table.txt
#########################################################################
def handler(signum, frame):
    agent.save(model_name, outdir)
    rospy.logwarn('Saved model before break this training process')
    exit(1)

signal.signal(signal.SIGINT, handler)
##########################################################################

if __name__ == '__main__':
    rospy.init_node('turtlebot3_world_qlearn', anonymous=True, log_level=rospy.WARN)
    model_name = "turtlebot3_v4_"
    
    # num_laser = 6
    # n_observations = num_laser * 2 +  2 # prev_laser, curr_laser, remain_dist, remain_heading
    # n_actions = 5
    num_laser = 24
    n_observations = num_laser +  2 # prev_laser, curr_laser, remain_dist, remain_heading
    n_actions = 5

    # Init OpenAI_ROS ENV
    environment_name= task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # alpha = rospy.get_param("/turtlebot3/alpha")
    alpha = 0.00025
    alpha_decay= 0.01
    batch_size = 64
    # epsilon = rospy.get_param("/turtlebot3/epsilon")
    # gamma = rospy.get_param("/turtlebot3/gamma")
    gamma = .99
    # epsilon_discount = rospy.get_param("/turtlebot3/epsilon_discount")
    # nepisodes = rospy.get_param("/turtlebot3/nepisodes")
    # nsteps = rospy.get_param("/turtlebot3/nsteps")
    max_env_steps= 500

    # running_step = rospy.get_param("/turtlebot3/running_step")
    
    rospackage_name = "my_turtlebot3_openai_example"
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    outdir = pkg_path + '/models'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        rospy.logfatal("Created folder="+str(outdir))

    agent = DQNSolver(  env,
                        n_observations = n_observations,
                        n_actions= env.action_space.n,
                        max_env_steps= max_env_steps,
                        gamma= gamma,
                        alpha= alpha,
                        alpha_decay= alpha_decay,
                        batch_size= batch_size, 
                        load_trained_model= True, 
                        load_episode=180)

    # agent.load(model_name, outdir)
    agent.run(num_episodes=100000, do_train=True)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    outdir = pkg_path + '/models'
    
    # agent.save(model_name, outdir)
    # agent.load(model_name, outdir)
    # agent.save(model_name, "./models")
    # agent.load(model_name, "./models")
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")
    # agent.run(num_episodes=10, do_train=False)
    env.close()