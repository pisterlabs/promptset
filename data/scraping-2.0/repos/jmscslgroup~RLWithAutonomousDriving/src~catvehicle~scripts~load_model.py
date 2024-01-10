#!/usr/bin/env python

import gym
import numpy as np
import time
import qlearn_hoang1
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from openai_ros.task_envs.catvehicle import catvehicle_wall

#Hoang import
from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            # For storing moves 
import random
from keras.models import load_model      # Load model

# for any function that calls start_qlearning.py, this will be the main
if __name__ == '__main__':

    rospy.init_node('catvehicle_wall_qlearn', anonymous=True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('CatVehicleWall-v0') # loads environment that was created in turtlebot2_maze.py
    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('catvehicle_openai_ros')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")
    
    nepisodes = rospy.get_param("/catvehicle/nepisodes")
    nsteps = rospy.get_param("/catvehicle/nsteps")
    gamma = rospy.get_param("/catvehicle/gamma")
    LEARNING_RATE = rospy.get_param("/catvehicle/LEARNING_RATE")
    MEMORY_SIZE = rospy.get_param("/catvehicle/MEMORY_SIZE")
    EXPLORATION_MAX = rospy.get_param("/catvehicle/EXPLORATION_MAX")
    EXPLORATION_MIN = rospy.get_param("/catvehicle/EXPLORATION_MIN")
    EXPLORATION_DECAY = rospy.get_param("/catvehicle/EXPLORATION_DECAY")
    BATCH_SIZE = rospy.get_param("/catvehicle/BATCH_SIZE")
    
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    #deepQlearning = qlearn_hoang1.QLearn(gamma,LEARNING_RATE,MEMORY_SIZE,EXPLORATION_MAX,EXPLORATION_MIN,EXPLORATION_DECAY,BATCH_SIZE,  observation_space, action_space)
    run = 0
    model = load_model('/home/reu-cat/catvehicle_ws/catVehicle_model.h5')
    
    for x in range(nepisodes):
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        cumulative_reward = 0
        for i in range(nsteps):
            action = np.argmax(model.predict(state)[0])
            state_next, reward, terminal, info = env.step(action)
            #reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            #deepQlearning.remember(state, action, reward, state_next, terminal)
            state = state_next
            cumulative_reward += reward
            if terminal:
                print "Run: " + str(run) + ", score: " + str(cumulative_reward)
                #score_logger.add_score(step, run)
                break
            #deepQlearning.experience_replay()
            
        #deepQlearning.save()
         
        
   
    
    

