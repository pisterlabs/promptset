'''
Proximial Policy Optimization (PPO)  Test Script for Aliengo Robot
Simulation will have 100 iteration and will show each reward-episode graph
Garen Haddeler
12.11.2020
'''
#python packages
import random
import time
import math
import matplotlib.pyplot as plt
#RL packages
import gym
import numpy as np
import tensorflow as tf
from ppo import PPOTrain,Policy_net
#Open-ai ROS packages
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
import rospy
import rospkg
import roslib
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelConfiguration

ITERATION = 100
GAMMA = 0.95

rospy.init_node('ppo_aliengo_test_node')

def main():

    LoadYamlFileParamsTest(rospackage_name="learning_ros", rel_path_from_package_to_file="config", yaml_file_name="aliengo_stand.yaml")
    # Init OpenAI_ROS ENV
    env = StartOpenAI_ROS_Environment('AliengoStand-v0')
    time.sleep(3)
    Policy = Policy_net('policy',ob_space =3,act_space = 8 )
    Old_Policy = Policy_net('old_policy',ob_space =3, act_space = 8 )
    PPO = PPOTrain(Policy, Old_Policy, gamma=GAMMA)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'model_train/model_ppo.ckpt')
        obs = env.reset()
        reward = 0
        success_num = 0
        scores = []

        for iteration in range(ITERATION):  # episode
            observations = []
            actions = []
            v_preds = []
            rewards = []
            run_policy_steps = 0
            while  not rospy.is_shutdown():  # run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=False)
                #print('act: ',act, 'v_pred: ',v_pred )
                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)
                next_obs, reward, done, _= env.step(act)  
                time.sleep(0.01)
                if done:
                    print(reward)
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    obs = env.reset()
                    break
                else:
                    obs = next_obs

            scores.append(sum(rewards)) 
        
            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, [len(observations), 3])
            actions = np.array(actions).astype(dtype=np.int32)
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) 

            PPO.assign_policy_parameters()

            inp = [observations, actions, rewards, v_preds_next, gaes]

        plt.plot(scores)
        plt.show()
        env.stop()


if __name__ == '__main__':
    main()
