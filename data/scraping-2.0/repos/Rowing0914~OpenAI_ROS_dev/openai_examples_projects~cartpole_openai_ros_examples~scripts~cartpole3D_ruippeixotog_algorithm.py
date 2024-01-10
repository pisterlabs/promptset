#!/usr/bin/env python
import rospy

import os

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from gym_runner import GymRunner
from q_learning_agent import QLearningAgent

# import our training environment
from openai_ros.task_envs.cartpole_stay_up import stay_up

class CartPoleAgent(QLearningAgent):
    def __init__(self):
        
        state_size = rospy.get_param('/cartpole_v0/state_size')
        action_size = rospy.get_param('/cartpole_v0/n_actions')
        gamma = rospy.get_param('/cartpole_v0/gamma')
        epsilon = rospy.get_param('/cartpole_v0/epsilon')
        epsilon_decay = rospy.get_param('/cartpole_v0/epsilon_decay')
        epsilon_min = rospy.get_param('/cartpole_v0/epsilon_min')
        batch_size = rospy.get_param('/cartpole_v0/batch_size')
        
        
        
        QLearningAgent.__init__(self,
                                state_size=state_size,
                                action_size=action_size,
                                gamma=gamma,
                                epsilon=epsilon,
                                epsilon_decay=epsilon_decay,
                                epsilon_min=epsilon_min,
                                batch_size=batch_size)
        
        #super(CartPoleAgent, self).__init__(4, 2)

    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu', input_dim=4))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(2))
        model.compile(Adam(lr=0.001), 'mse')

        # load the weights of the model if reusing previous training session
        # model.load_weights("models/cartpole-v0.h5")
        return model


if __name__ == "__main__":
    rospy.init_node('cartpole3D_ruippeixotog_algorithm', anonymous=True, log_level=rospy.FATAL)
    
    episodes_training = rospy.get_param('/cartpole_v0/episodes_training')
    episodes_running = rospy.get_param('/cartpole_v0/episodes_running')
    max_timesteps = rospy.get_param('/cartpole_v0/max_timesteps', 10000)
    
    gym = GymRunner('CartPoleStayUp-v0', 'gymresults/cartpole-v0', max_timesteps)
    agent = CartPoleAgent()

    gym.train(agent, episodes_training, do_train=True, do_render=False, publish_reward=False)
    gym.run(agent, episodes_running, do_train=False, do_render=False, publish_reward=False)

    agent.model.save_weights("models/cartpole-v0.h5", overwrite=True)
    #gym.close_and_upload(os.environ['API_KEY'])