# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:46:43 2020

@author: Kata
"""
"""
Inspired by:
    https://pythonprogramming.net/convolutional-neural-network-deep-learning-python-tensorflow-keras/
    https://github.com/fakemonk1/Reinforcement-Learning-Lunar_Lander/blob/master/Lunar_Lander.py
    Jonas' algorithm
"""


# Import the gym environment from OpenAI
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# Sequential NN model is the most common one.
from keras import Sequential
# Layers used for NNs: Conv2D is usually used for image recognition,
# Dense is commonly used, but may be prone to overfitting.
from keras.layers import Dense, Conv2D
# Allows using functions such as Flatten* (when trying to change from Conv2D to Dense layer)
# or MaxPooling, which is used in Conv2D layers.
# * Flatten converts 3D feature maps (Conv2D) into 1D feature vectors
# from keras import Flatten, MaxPooling2D, Dropout
# Activation functions: relu (rectified linear) is standard in NN
# linear is used for the final layer to get just one possible answer.
from keras.activations import relu, linear
# Standard optimizer is adam.
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import load_model

# Using these parameters in nested loops would allow for creating different combinations of NNs and find the best combination
num_deep_layers = [1, 2, 3]
num_neurons = [32, 64, 128]


class DQN:
    def __init__(self, env, lr, gamma, epsilon, epsilon_decay, deep_layers, neurons):
        
        # Initializes variables based on the environment (e.g. LunarLander-V2, Cartpole-V0, etc.)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_action_space = self.action_space.n
        self.num_observation_space = env.observation_space.shape[0]
        self.episodes_rewards = []
        self.average_episodes_rewards = []
        self.trained_episodes_rewards = []
        self.average_trained_episodes_rewards = []
        
        # Keep track of how many frames the model ran through in total.
        self.training_total_frame_count = 0
        
        # Initializes variables based on the hyperparameters given.
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # Initializes variables the same every time, as given below.
        self.replay_memory_buffer = deque(maxlen = 500_000)
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.counter = 0
        
        # Enables initalising NNs with multiple deep layers at varying size.
        self.deep_layers = deep_layers
        self.neurons = neurons
        self.name = "{}_Deep_Layers-{}_Neurons-Timestamp_{}".format(self.deep_layers, self.neurons, int(time.time()))
        
        
        self.model = self.initialize_model()
        
        
    
        
        

    # Constructs model using sequential model and different (deep) layers.
    def initialize_model(self):
        model = Sequential()
        
        
        # Input layer (based on observation space of the environment)
        model.add(Dense(2*self.neurons, input_dim = self.num_observation_space, activation = relu))
        # model.add(Dropout(0.2))
        
        # Deep layers
        for i in range(self.deep_layers):
            model.add(Dense(self.neurons, activation = relu))
        
        # Output layer (based on action space of the environment)
        model.add(Dense(self.num_action_space, activation = linear))
        
        
        # Compile the model giving the loss and the optimizer as an argument.
        model.compile(loss = mean_squared_error, optimizer = Adam (lr = self.lr))
        
        # Prints out the stats of the model to give an overview over what was just created.
        print(self.name)
        print(model.summary())
        
        
        return model

    # Decide whether to take an exploratory or exploitative action.
    def get_action(self, state):
        
        # Based on a random number 0 <= n <= 1, if n smaller than the current epsilon e, select random action based on the action space of the environment.
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_action_space)

        # Otherwise let the model decide the best action in the current environment state based on the momentary policy.
        predicted_actions = self.model.predict(state)
        
        # Return the action to be taken in the current state.
        return np.argmax(predicted_actions[0])
    
    def learn_and_update_weights_by_reply(self):
        
        # replay_memory_buffer size check (Needs rewording / more understanding)
        if len(self.replay_memory_buffer) < self.batch_size or self.counter != 0:
            return
        
        # If the model has been completing the task with a desirable reward for a while, stop it to prevent it from overfitting.
        if len(self.episodes_rewards) > 10:
            if np.mean(self.episodes_rewards[-10]) > 180:
                return
        
        random_sample = self.get_random_sample_from_replay_mem()
        
        # Convert the chosen experience's attributes to the needed parameters (state, action, etc.)
        states, actions, episodes_rewards, next_states, done_list = self.get_attributes_from_sample(random_sample)
        
        # Update the episodes_rewards based on the discount factor (gamma) influencing the next set of states. (Needs revising / more understanding)
        targets = episodes_rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis = 1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets
        
        # Adjusts the policy based on states, target vectors and other things (needs more understanding)
        self.model.fit(states, target_vec, epochs = 1, verbose = 0)
    
    def add_to_replay_memory(self, state, received_action, reward, next_state, done):
        self.replay_memory_buffer.append((state, received_action, reward, next_state, done))
    
    # Choose a random past experience from the replay memory
    def get_random_sample_from_replay_mem(self):
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample
        
    def get_attributes_from_sample(self, random_sample):
        
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        episodes_rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        return np.squeeze(states), actions, episodes_rewards, next_states, done_list
     
    
    def train(self, num_episodes):
        
        for episode in range(num_episodes):
            
            state = env.reset()
            episode_reward = 0
            episode_frame_count = 0
            num_steps = 1000
            state = np.reshape(state, [1, self.num_observation_space])
            
            for step in range(num_steps):
                
                # env.render()
                
                # Decide what action to take.
                received_action = self.get_action(state)
                
                # Step to next environment state with current environment state tuple.
                next_state, reward, done, info = env.step(received_action)
                
                # Update the next state based on the tuple and the observation space.
                next_state = np.reshape(next_state, [1, self.num_observation_space])
                
                # Add the experience of the state-action pair to the replay memory
                self.add_to_replay_memory(state, received_action, reward, next_state, done)
                
                # Add the reward for this step to the episode reward
                episode_reward += reward
                
                # Progress to the next state by changing the current state to become the next state.
                state = next_state
                
                # Update counter used in the replay memory buffer size check.
                self.update_counter()
                
                # Update the weight connections within the layers.
                self.learn_and_update_weights_by_reply()
                
                self.training_total_frame_count += 1
                episode_frame_count += 1
                
                
                if done:
                    break
                
            # Add the episode reward to the list of episodes_rewards for the episodes    
            self.episodes_rewards.append(episode_reward)
            
            # Reduce the epsilon based on decay rate to move the focus of the NN from exploration to exploitation over time. 
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Stop if the model has solved the environment (reward must average above 200).
            if len(self.episodes_rewards) > 100:
                self.average_episodes_rewards = np.mean(self.episodes_rewards[-100])
            else:
                self.average_episodes_rewards = np.mean(self.episodes_rewards)
            if self.average_episodes_rewards > 200:
                print("DQN Training Complete...")
                break
            
            # Print out the episode's results with additional information.
            print("""Episode: {}\t\t\t|| Episode Reward: {:.2f}
Last Frame Reward: {:.2f}\t|| Average Reward: {:.2f}\t|| Epsilon: {:.2f}
Frames this episode: {}\t\t|| Total Frames trained: {}\n"""
                  .format(episode, episode_reward, reward, self.average_episodes_rewards, 
                          self.epsilon, episode_frame_count, 
                          self.training_total_frame_count))
            # print("Episode: ", episode, "\t\t\t\t\t|| Episode Reward:", episode_reward,
            #       "\nLast frame Reward: ", reward, "\t\t|| Average Reward: ", self.average_episodes_rewards, "\t|| Epsilon: ", self.epsilon,
            #       "\nTotal Frames trained: ", self.training_total_frame_count, "\t|| Frames this episode: ", episode_frame_count)
    
            
            if episode % 50 == 0:
                plt.plot(self.average_episodes_rewards)
                plt.plot(self.episodes_rewards)
                title = "DQN Training Curve: \n"
                title += self.name
                plt.title(self.name)
                plt.xlabel("Episode")
                plt.ylabel("Rewards")
                plt.show()
                
                
        env.close()
        figname = "Figure_"
        figname += self.name
        plt.savefig(figname)
    
    
    
    
    
    
    # Counter used for experience replay.
    def update_counter(self):
        self.counter += 1
        step_size = 5
        self.counter = self.counter % step_size
        
    def save(self):
        self.model.save(self.name)
    

# Makes a validation run of a trained model, which is very similar to a training run.
def test_trained_model(self, trained_model, num_episodes):
    
    # episodes_rewards_list = []
    print("Start validation run of trained model:")
    
    num_steps = 1000
    
    for episode in range(num_episodes):
        current_state = env.reset()
        num_observation_space = env.observation_space.shape[0]
        current_state = np.reshape(current_state, [1, num_observation_space])
        episode_reward = 0
        
        for step in range(num_steps):
            env.render()
            selected_action = np.argmax(trained_model.predict(current_state)[0])
            new_state, reward, done, info = env.step(selected_action)
            new_state = np.reshape(new_state, [1, num_observation_space])
            current_state = new_state
            episode_reward += reward
            
            if done:
                break
        
        if len(self.trained_episodes_rewards) > 100:
            average_trained_episodes_rewards = np.mean(self.trained_episodes_rewards[-100:])
        else:
            average_trained_episodes_rewards = np.mean(self.trained_episodes_rewards)
        self.trained_episodes_rewards.append(episode_reward)
        
        print("""Episode: {}\t\t\t|| Episode Reward: {:.2f}\
Last Frame Reward: {:.2f}\t|| Average Reward: {:.2f}"""
              .format(episode, episode_reward, reward, average_trained_episodes_rewards))
        
    # return episodes_rewards_list
    env.close()
    plt.plot(self.average_trained_episodes_rewards)
    plt.plot(self.trained_episodes_rewards)
    title = "Testing for trained DQN: \n"
    title += self.name
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.show()

        
if __name__ == '__main__':

    # Create one of the environments from OpenAI
    env = gym.make("LunarLander-v2")
    env.reset
    
    # Create random seeds (Elaborate on what this does.)
    env.seed(21)
    np.random.seed(21)
    
    # Initialize hyper-parameters
    lr = 0.001
    eps = 1.0
    eps_decay = 0.995
    gamma = 0.99
    training_episodes = 201
    
    
    # Allows for comparison between different models.
    for deep_layers in num_deep_layers:
        for neurons in num_neurons:
            # name = "{}_Deep_Layers_{}_Neurons_Timestamp_{}".format(deep_layers, neurons, int(time.time()))
            # print("Training model: " + name)
            # print(model.summary())
            model = DQN(env, lr, gamma, eps, eps_decay, deep_layers, neurons)
            model.train(training_episodes)
            
    #         # Continuously train the model until it reaches the target average reward.
    #         while (np.mean(model.episodes_rewards[-10:]) < 180):
    #             model.train(training_episodes)
    #         model.save(name)
            
    
    
    # model = DQN(env, lr, gamma, eps, eps_decay)
    # model.train(training_episodes)
    # # Continuously train the model until it reaches the target average reward.
    # while (np.mean(model.episodes_rewards[-10:]) < 180):
    #     model.train(training_episodes)
    
    
    # trained_model = load_model("replay_DQN_trained_model3.h5")
    # model.test_trained_model(trained_model, num_episodes=30)

    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        