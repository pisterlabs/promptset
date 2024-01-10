#!/usr/bin/env python3

from functools import reduce
import numpy as np
import time
import cv2
import torch
from torchvision import transforms
# ROS packages required
import rospy
import rospkg

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from agents.comb import DCAE_DDPG
from utils import set_seed

def state_resize(state, width, height):
    img = state["image"]
    new_img = cv2.resize(img, (width, height))
    state["image"] = new_img
    return state

if __name__ == '__main__':

    rospy.init_node('mycobot_RL', anonymous=True, log_level=rospy.DEBUG)
    
    set_seed(0)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param('/mycobot/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('mycobot_training')

    last_time_steps = np.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    gamma = rospy.get_param("/mycobot/gamma")
    batch_size = rospy.get_param("/mycobot/batch_size")
    memory_size = rospy.get_param("/mycobot/memory_size")
    nepisodes = rospy.get_param("/mycobot/nepisodes")
    nsteps = rospy.get_param("/mycobot/nsteps")

    running_step = rospy.get_param("/mycobot/running_step")
    
    img_width = rospy.get_param("/mycobot/width")
    img_height = rospy.get_param("/mycobot/height")
    
    hidden_dim = rospy.get_param("/mycobot/hidden_dim")

    # Initialises the algorithm that we are going to use for learning
    img_size = (img_height, img_width, 4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    agent = DCAE_DDPG(img_size, hidden_dim, env.observation_space, 
                      env.action_space, gamma, batch_size=batch_size, 
                      memory_size=memory_size, device=device)
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    rospy.logwarn("Start Data Collection")   
    state = env.reset()
    for _ in range(memory_size):
        # state["image"] = trans(state["image"]).numpy()
        state = state_resize(state, img_width, img_height)
        action = env.action_space.sample()
        next_state, reward, success, done, _ = env.step(action)
        # next_state["image"] = trans(next_state["image"]).numpy()
        next_state = state_resize(next_state, img_width, img_height)
        transition = {
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'action': action,
            'success': int(success),
            'done': int(done)
        }
        agent.ddpg.replay_buffer.append(transition)
        state = env.reset() if success or done else next_state
    rospy.logwarn("Data Collected")    
        
    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do
    # for x in range(nepisodes):
    #     rospy.logdebug("############### WALL START EPISODE=>" + str(x))

    #     cumulated_reward = 0
    #     done = False

    #     # Initialize the environment and get first state of the robot
    #     state = env.reset()

    #     # for each episode, we test the robot for nsteps
    #     for i in range(nsteps):
    #         state = state_resize(state, img_width, img_height)
    #         rospy.logwarn("############### Start Step=>" + str(i))
    #         # Pick an action based on the current state
    #         action = agent.get_action(state)
    #         # Execute the action in the environment and get feedback
    #         next_state, reward, success, done, _ = env.step(action)
    #         next_state = state_resize(next_state, img_width, img_height)
    #         transition = {
    #             'state': state,
    #             'next_state': next_state,
    #             'reward': reward,
    #             'action': action,
    #             'success': int(success),
    #             'done': int(done)
    #         }
    #         agent.ddpg.replay_buffer.append(transition)

    #         cumulated_reward += reward
    #         if highest_reward < cumulated_reward:
    #             highest_reward = cumulated_reward

    #         # Make the algorithm learn based on the results
    #         rospy.logwarn("# reward that action gave=>" + str(reward))
    #         rospy.logwarn("# episode cumulated_reward=>" + str(cumulated_reward))
    #         agent.update()

    #         if not (done):
    #             rospy.logwarn("NOT DONE")
    #             state = next_state
    #         else:
    #             rospy.logwarn("DONE")
    #             last_time_steps = np.append(last_time_steps, [int(i + 1)])
    #             break
    #         rospy.logwarn("############### END Step=>" + str(i))
    #         #raw_input("Next Step...PRESS KEY")
    #         # rospy.sleep(2.0)
    #     m, s = divmod(int(time.time() - start_time), 60)
    #     h, m = divmod(m, 60)
    #     rospy.logerr(("EP: " + str(x + 1) + " - Reward: " + str(cumulated_reward) + " Time: %d:%02d:%02d" % (h, m, s)))

    # rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(highest_reward) + "| PICTURE |"))

    # l = last_time_steps.tolist()
    # l.sort()

    # # print("Parameters: a="+str)
    # rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    # rospy.loginfo("Best 100 score: {:0.2f}".format(
    #     reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
