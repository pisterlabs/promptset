#!/usr/bin/env python3

import rospy                                                        ###
from openai_ros.task_envs.sawyer import learn_to_touch_cube         ###

import gym                                              			###
# import sys                                                          ###
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')     ###
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env

from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')     ###

# import filter_env

import gc
gc.enable()

###################################################################################################

ENV_NAME ='SawyerTouchCube-v0'                                      ###
EPISODES = 100000
TEST = 10

def main():
    rospy.init_node('sawyer_learn_to_pick_cube_a2c', anonymous=True, log_level=rospy.WARN)     ###

    # env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    # env = gym.make(ENV_NAME)   

    # Parallel environments
    env = make_vec_env(ENV_NAME, n_envs=4)              			###
    # env = VecFrameStack(ENV_NAME, 4)								###
    model = A2C(MlpPolicy, env, verbose=1)            				###  
    # model = A2C(CnnPolicy, env, lr_schedule='constant')			###

    env = gym.wrappers.Monitor(env, '~/catkin_ws/src/A2C/experiments/' + ENV_NAME,force=True)

    model.learn(total_timesteps=25000)                  			###
    model.save("a2c_sawyer")

    for episode in xrange(EPISODES):
        state = env.reset()
        # Train
        for step in xrange(env.spec.timestep_limit):

            action = agent.noise_action(state)    
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            for i in xrange(TEST):
                state = env.reset()
                for j in xrange(env.spec.timestep_limit):
					#env.render()
                    action = agent.action(state) # direct action for test ###TO EDIT: check chooseAction my_sawyer_openai_example/scripts/qlearn.py
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('episode: ',episode,'Evaluation Average Reward:',ave_reward)
    env.monitor.close()

if __name__ == '__main__':
    main()
