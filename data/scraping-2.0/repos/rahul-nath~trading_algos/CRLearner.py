"""
(c) 2017 Rahul Nath
CRLearner for continuous state learning. Code adapted from OpenAI Gym examples
(https://gym.openai.com)

FeedForwardNeuralNetwork borrowed from same source.
"""

import numpy as np
import random as rand
from FFNN import FeedForwardNeuralNetwork
from collections import deque

class CRLearner(object):

    def __init__(self, \
        num_dimensions = 2, \
        num_actions = 4, \
        verbose = False):
        self.num_dimensions = num_dimensions
        self.num_actions = num_actions
        self.done = False
        self.verbose = verbose
        self.network = FeedForwardNeuralNetwork([num_dimensions,  13*num_dimensions, 13*num_dimensions, num_actions])
        self.episodes = 0 

        # Hyperparameters of the training
        self._GAMMA = 0.99    
        self._TRAINING_PER_STAGE = 7
        self._MINIBATCH_SIZE = 32    
        self._REPLAY_MEMORY = 5000    

        # Exploration/exploitations parameters
        self._epsilon = .80
        self._EPSILON_DECAY = 0.99
        self._EPISODES_PURE_EXPLORATION = 1
        self._MIN_EPSILON = 0.01

        # Define useful variables
        self.a = np.zeros(self.num_actions)
        self._replay_mem, self.s = deque(), None


    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.episodes = 0    
        if self.a.sum() == 0:
            action = rand.randint(0, self.num_actions-1)
        else:
            value_per_action = self.network.predict(s)
            action = np.argmax(value_per_action)  
            self.a = np.zeros(self.num_actions)
            self.a[action] = 1
        if self.verbose: print "s =", s,"a =", action
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        self.episodes += 1
        if r == 1:
            self.done = True
        else:
            self.done = False

        if self.done:
            self._epsilon = max(self._epsilon * self._EPSILON_DECAY, self._MIN_EPSILON)       

        # Reshape the data
        current_state = s_prime.reshape((1, len(s_prime)))

        if self.s is None:
            self.s = current_state
            value_per_action = self.network.predict(self.s)
            action = np.argmax(value_per_action)  
            self.a = np.zeros(self.num_actions)
            self.a[action] = 1
            return action

        # Store the last transition
        new_decision= [self.s.copy(), 
                            self.a.copy(),
                            r,
                            current_state.copy(),
                            self.done]
        self._replay_mem.append(new_decision)
        self.s = current_state.copy()

        # If the memory is full, pop the oldest stored transition
        while len(self._replay_mem) >= self._REPLAY_MEMORY:
            self._replay_mem.popleft()


        # Only train and decide after enough episodes of random play
        if self.episodes > self._EPISODES_PURE_EXPLORATION:
  
            for _ in range(self._TRAINING_PER_STAGE):
                # Sample a mini_batch to train on
                # names correspond to index in lists holding the mentioned value
                batches = np.random.randint(len(self._replay_mem), size=len(self._replay_mem))[:self._MINIBATCH_SIZE]
                prev_states = np.vstack([self._replay_mem[i][0] for i in batches])
                actions = np.vstack([[self._replay_mem[i][1]] for i in batches])
                current_states = np.vstack([self._replay_mem[i][3] for i in batches])
                rewards = np.array([self._replay_mem[i][2] for i in batches]).astype('float')        
                done = np.array([self._replay_mem[i][4] for i in batches]).astype('bool')

                # Calculates the value of the current_states (per action)
                pred_actions = self.network.predict(current_states)
                
                # Calculate the empirical target value for the previous_states
                targets = rewards + ((1. - done) * self._GAMMA * pred_actions.max(axis=1))

                # Run a training step
                self.network.fit(prev_states, actions, targets)      
                
            if np.random.random() > self._epsilon:
                action_values = self.network.predict(self.s)
                action = np.argmax(action_values)  
            else:
                action = np.random.randint(0, self.num_actions)

        else:
            action = np.random.randint(0, self.num_actions)


        next_action = np.zeros([self.num_actions])
        next_action[action] = 1
        self.a = next_action
          
        return action
