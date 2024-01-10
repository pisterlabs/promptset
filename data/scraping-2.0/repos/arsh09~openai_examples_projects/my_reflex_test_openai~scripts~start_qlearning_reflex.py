#!/usr/bin/env python2

import sys
# if sys.version_info[0] < 3:
#         raise Exception("Must be using Python 3 on ROS")

# import tensorflow as tf
import gym
import numpy as np
import time
# import qlearn
import random
from gym import wrappers
from gym.envs.registration import register
# ROS packages required
import rospy
import rospkg
# import our training environment
# from openai_ros.task_envs.iiwa_tasks import iiwa_move
# from openai_ros.task_envs.hopper import hopper_stay_up
# import pickle, os

# from baselines import PPO2
# from run_algo import start_learning
import subprocess

# For the launch 
import roslaunch
import os

import git
import sys
# save part
import pickle
import matplotlib.pyplot as plt
import termios, tty # for keyboard

# Global variables:
here = os.path.dirname(os.path.abspath(__file__))

# Functions
def random_action():
    a = []
    for i in range(0, 6):
        a.append(random.uniform(-0.1, 0.1))
        # i = i+1
    a[2]=-0.05
    return a

def defined_action():
    a = []
    for i in range(0, 6):
        if i == 0 or 1 or 2:
            a.append(-0.01)
        else:
            a.append(0.0)
        # i = i+1
    # a[2]=-0.05
    return a

def discrete_action(action):
    '''
    Transform the asking action to a discretize action to simplify the problem 
    0 = + step_size * x
    1 = - step_size * x
    2 = + step_size * y
    3 = + step_size * y
    4 = + step_size * z
    5 = - step_size * z
    '''
    a = [0, 0, 0, 0, 0, 0]
    step_size = 0.01
    if action == 0:
        a[action] = step_size
    elif action == 1:
        a[0] = -step_size
    elif action == 2:
        a[1] = step_size
    elif action == 3:
        a[1] = -step_size
    elif action == 4:
        a[2] = step_size
    elif action == 5:
        a[2] = -step_size
    return a

# Transfer the position of the robot to a state allowing q-learning with q table
def env_to_state(x, y, z):
    state = 0
    x = (int(10*x) + 8)
    y = (int(10*y) + 8) * 17
    z = (int(10*z)) * 289
    state = x+y+z
    return state

# save the data
def save(list_demos, name="demos_default_name"):
    '''
    Save a list in a .pkl file on the harddisk. Return if it did it or not.
    '''
    global here
    saved = False
    try:
        name = name + ".pkl"
        with open(os.path.join(here, name), 'wb') as f:
            pickle.dump(list_demos, f, protocol=pickle.HIGHEST_PROTOCOL)
        saved = True
    except:
        print("ERROR: Couldn't save the file .pkl")
    return saved

def save_txt(number_dones):
    '''
    Save a sumary in a txt files
    '''
    name = "saves/dones.txt"
    file = open(os.path.join(here, name),'w')
    file.write("##############################################################")
    file.write("######################  Summary  #############################")
    file.write("##############################################################")
    file.write("Number of done: ")
    file.write(str(number_dones))
    file.write("\n")
    
    file.close() 


# Load the data
# TODO
def load(name="demos_default_name"):
    '''
    Load the needed data
    '''
    name = name + ".pkl"
    with open(name, 'rb') as f:
        return pickle.load(f)

def init_env():
    '''
    Init the environment
    '''
    # Cheating with the registration 
    # (shortcut: look at openai_ros_common.py and task_envs_list.py)
    timestep_limit_per_episode = 10000
    register(
        id="iiwaMoveEnv-v0",
        entry_point='openai_ros.task_envs.iiwa_tasks.iiwa_move:iiwaMoveEnv',
        max_episode_steps=timestep_limit_per_episode,
    )

    # Create the Gym environment
    env = gym.make('iiwaMoveEnv-v0')
    print("Gym environment done")
    return env

# Using continue actions
def test_function_cont(env, max_steps):
    # Where I test EVERYTHING
    # observation, reward, done, info
    for i in range(0, max_steps):
        # raw_input("Press Enter to continue...")
        # a=random_action()
        a = defined_action()
        # env.step(a)
        observation, reward, done, info = env.step(a)
        print("*********************************************")
        print("Observation: ", observation)
        print("Reward: ", reward)
        print("Done: ", done)
        print("Info: ", info)
        print("Action: ",  a)
        print("*********************************************")

# Test discrete action
def test_function_discrete(env, max_steps):
    for i in range(0, max_steps):
        action = env.action_space.sample()
        # print(action)
        discrete_action_vector = discrete_action(action)
        observation, reward, done, info = env.step(discrete_action_vector)
        print("*********************************************")
        print("Observation: ", observation)
        print("Reward: ", reward)
        print("Done: ", done)
        print("Info: ", info)
        print("Action: ",  action)
        print("*********************************************")

def plot2d(increment, reward):
    '''
    Reward history. 
    '''
    print("plotting")

    fig = plt.figure()
    plt.plot(increment, reward)
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.title('Reward over time')
    plt.savefig('reward.jpg')     
    plt.show()
    # time.sleep(5.0)
    plt.close(fig)

def qlearning(env):
    
    # General Parameters
    max_episode = 100
    max_steps = 25
    MAX_EPISODES = max_episode # Number of episode

    action_size = env.action_space.n
    state_size = 4046
    bool_save = True
    
    # # Q = np.zeros([env.observation_space.n,env.action_space.n])
    Q_table = np.zeros((state_size, action_size))
    
    # print Q_table
    # Table with the reward and episode for ploting
    rewards_list = []
    episode_list = []

    # Parameters qlearning
    ALPHA = 0.8
    GAMMA = 0.95   

    EPSILON = 1.0
    MAX_EPSILON = 1.0
    MIN_EPSILON = 0.01
    DECAY_RATE = 0.005
    done_increment = 0
    # time_start = rospy.Time.now()
    for episode in range(MAX_EPISODES):
        # time_episode_start = rospy.Time.now()

        # Save the different data
        if episode % 20 == 0 and bool_save:
            save(Q_table, name=("saves/qtable_qlearning_"+str(episode)))
            save(rewards_list, name=("saves/rewards_list_qlearning_"+str(episode)))
            save(episode_list, name=("saves/episode_list_qlearning_"+str(episode)))
            save_txt(done_increment)
            # plot2d(episode_list, rewards_list)
            # print("FILE SAVED!!!")
            
        observation = env.reset()
        # rospy.sleep(5.0)

        # To be sure we can save reset the env while the robot is moving
        # rospy.sleep(1.0)
        state = env_to_state(observation[0], observation[1], observation[2])

        # Counter an sum to reset
        step = 0
        done = False
        total_rewards = 0
        
        # Loop where the robot move and learn
        while (not done) and (step <= max_steps):
    
            if random.uniform(0, 1) < EPSILON:
                discrete_act = env.action_space.sample()
            else:
                discrete_act = np.argmax(Q_table[state, :])
            
            # Convert the action to discrete action
            action = discrete_action(discrete_act)
            # Do the step in the world
            new_observation, reward, done, info = env.step(action)
            # New observation to state
            new_state = env_to_state(new_observation[0], new_observation[1], new_observation[2])
            print("*********************************************")
            print("Observation: ", observation)
            print("State: ", state)
            print("Reward: ", reward)
            print("Done: ", done)
            print("# dones: ", done_increment)
            print("Info: ", info)
            print("Action: ",  action)
            print("Episode: ", episode)
            print("Step: ", step)
            print("*********************************************")

            # Q calulation
            q_predict = Q_table[state, discrete_act]

            if done:
                q_target = reward
                done_increment +=1
            else:
                q_target = reward + GAMMA * np.max(Q_table[new_state, :])
            Q_table[state, discrete_act] +=  ALPHA * (q_target - q_predict)

            # Update the observation, reward, step
            observation = new_observation
            state = new_state
            total_rewards += reward
            step += 1

            # rospy.sleep(0.1)
        #End of the robot movement (reset the world for different reason done or max_step reached)

        EPSILON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)
        # print(EPSILON)
        rewards_list.append(total_rewards)
        episode_list.append(episode)
        
    # End of the qlearning

# Main function:
def main():
    # Begining of the script
    print("Python version: ", sys.version)
    rospy.init_node('start_qlearning_reflex', anonymous=True, log_level=rospy.WARN)


    env = init_env()
    env.reset()
    # save_txt(5)
    # qlearning(env)

    # env.reset()
    # rospy.sleep(3.0)
    # # Move by catching
    # env.step([0.5, 0.0, 0.25, 3.14, 0.0, 0.0])
    # rospy.sleep(15.0)
    # env.step([0.5, 0.0, 0.50, 3.14, 0.0, 0.0])
    # env.step([0.5, -0.2, 0.150, 3.14, 0.0, 0.0])

    # Move by sliding
    # env.step([0.5, 0.5, 0.15, 3.14, 0.0, 0.0])#1.57
    # env.step([0.5, 0.25, 0.15, 3.14, 0.0, 0.0])
    # env.step([0.5, 0.0, 0.150, 3.14, 0.0, 0.0])
    # env.step([0.5, -0.2, 0.150, 3.14, 0.0, 0.0])

    # print("between 2 actions")
    # env.step([0.5, 0.0, 0.2, 0.0, 0.0, 0.0])
    # print("between 2 actions")
    # env.reset()
    
    rospy.sleep(150.0)
    print("Close node: start_qlearning_reflex.py")
    print("Close node: start_qlearning_reflex.py")
    print("Close node: start_qlearning_reflex.py")
    print("Close node: start_qlearning_reflex.py")
    print("Close node: start_qlearning_reflex.py")
    print("Close node: start_qlearning_reflex.py")
    print("Close node: start_qlearning_reflex.py")
    # env.close()

if __name__ == '__main__':
    main()

# Test functions:
def hystory_test_function(env):
    # # tf.__version__
    # # Parameters
    # timestep_limit_per_episode = 10000
    # max_episode = 600
    # max_steps = 100


    # # print(env_to_state(0.8, 0.8, 1.3))
    # # Cheating with the registration 
    # # (shortcut: look at openai_ros_common.py and task_envs_list.py)
    # register(
    #     id="iiwaMoveEnv-v0",
    #     entry_point='openai_ros.task_envs.iiwa_tasks.iiwa_move:iiwaMoveEnv',
    # #     timestep_limit=timestep_limit_per_episode, #old one...
    #     max_episode_steps=timestep_limit_per_episode,
    # )

    # # Create the Gym environment
    # env = gym.make('iiwaMoveEnv-v0')
    # # rospy.loginfo("Gym environment done")
    # print("Gym environment done")
    # print(tf.__version__) # test for tensorflow
    # # state_size = env.observation_space.n
    # # action_size = env.action_space.n
    # # print("Ovservation space: ", state_size)
    # # print("Action space: ", action_size)

    # # rospy.sleep(10.0)
    # # print("Before reset the env" )
    # env.reset()
    # # print("After reset the env" )

    # action = env.action_space.sample()
    # print(action)
    # discrete_action_vector = discrete_action(action)
    # observation, reward, done, info = env.step(discrete_action_vector)
    # print("*********************************************")
    # print("Observation: ", observation)
    # print("Reward: ", reward)
    # print("Done: ", done)
    # print("Info: ", info)
    # print("Action: ",  action)
    # print("*********************************************")

    # rospy.sleep(3.0)

    # print("*********************************************")
    # print("Observation: ", observation)
    # print("Reward: ", reward)
    # print("Done: ", done)
    # print("Info: ", info)
    # print("Action: ",  action)
    # print("*********************************************")

    #TRY Q



    # print("Action 1" )
    # # action = [-0.1, -0.1, -0.1, 1.3, 0.1, 0.5]
    # action = [-0.1, -0.1, -0.1, 0.0, 0.0, 0.0]
    # observation, reward, done, info = env.step(action)
    # print("*********************************************")
    # print("Observation: ", observation)
    # print("Reward: ", reward)
    # print("Done: ", done)
    # print("Info: ", info)
    # print("Action: ",  action)
    # print("*********************************************")
    # # print("Set Action: " + str(action))
    # # env.step(action)


    # rospy.sleep(10.0)

    # print("Action 2" )
    # # action2 = [0.2, 0.2, 0.2, -1.3, -0.1, -0.5]
    # action2 = [0.3, -0.1, -0.2, 0.0, 0.0, 0.0]
    # observation, reward, done, info = env.step(action2)
    # print("*********************************************")
    # print("Observation: ", observation)
    # print("Reward: ", reward)
    # print("Done: ", done)
    # print("Info: ", info)
    # print("Action: ",  action2)
    # print("*********************************************")
    # # print("Set Action: " + str(action2))
    # # env.step(action2)

    # print("Action are sent")


    # print("Before reset the env" )
    # env.reset()
    # print("After reset the env" )
    
    
    
    # agent = DQNRobotSolver(environment_name,
    #                         n_observations,
    #                         n_actions,
    #                         n_win_ticks,
    #                         min_episodes,
    #                         max_env_steps,
    #                         gamma,
    #                         epsilon,
    #                         epsilon_min,
    #                         epsilon_log_decay,
    #                         alpha,
    #                         alpha_decay,
    #                         batch_size,
    #                         monitor,
    #                         quiet)
    # agent.run(num_episodes=n_episodes_training, do_train=True)
    
    

    # Define and train a model in one line of code !
    # trained_model = PPO2('MlpPolicy', 'CartPole-v1').learn(total_timesteps=10000)
    # you can then access the gym env using trained_model.get_env()
    
    
    
    
    # env._set_action(action)
    # # Set the logging system
    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('my_hopper_openai_example')
    # outdir = pkg_path + '/training_results'
    # env = wrappers.Monitor(env, outdir, force=True)
    # rospy.loginfo("Monitor Wrapper started")


    # # Where I test EVERYTHING
    #  # observation, reward, done, info
    # for i in range(0, 9):
    #     # raw_input("Press Enter to continue...")
    #     # a=random_action()
    #     a = defined_action()
    #     # env.step(a)
    #     observation, reward, done, info = env.step(a)
    #     print("*********************************************")
    #     print("Observation: ", observation)
    #     print("Reward: ", reward)
    #     print("Done: ", done)
    #     print("Info: ", info)
    #     print("Action: ",  a)
    #     print("*********************************************")

    # start_learning()
    # script = ["python3.6", "/home/roboticlab14/catkin_ws/src/openai_examples_projects/my_reflex_test_openai/scripts/run_algo.py"]    
    # process = subprocess.Popen(" ".join(script),
    #                                 shell=True 
    #                                 # env={"PYTHONPATH": "."}
    #                                 )

    # python3_command = ["python3.6", "/home/roboticlab14/catkin_ws/src/openai_examples_projects/my_reflex_test_openai/scripts/run_algo.py"]  # launch your python2 script using bash

    # process = subprocess.Popen(python3_command, stdout=subprocess.PIPE, shell=True)
    # output, error = process.communicate()  # receive output from the python2 script


    # print("Before reset the env" )
    # env.reset()
    # print("After reset the env" )

    # for i in range(0,10):
    #     a=random_action()
    #     env.step(a)
    #     print(a)

    # print("Before reset the env" )
    # env.reset()
    # print("After reset the env" )
    

    # To never finish
    # while True:
    #     a=1
        # a=random_action()
        # env.step(a)
        # print()

    # For testing 
    # for episode in range(max_episode):
    #     observation = env.reset()
    #     print(episode)
    print("Close node: start_qlearning_reflex.py")