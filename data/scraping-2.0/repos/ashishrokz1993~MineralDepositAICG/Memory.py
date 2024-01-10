'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
This module contains the classes for the dynamic memory to store the training data of RL and sample it when required
'''
import numpy as np
import globalVars as gv
from collections import namedtuple, deque
## Most of the code is from openAI github repository https://github.com/openai/baselines/tree/master/baselines/ddpg
class RingBuffer(object):
    '''
    This class holds the funcitons to fetch, append, and add the data to the memory
    '''
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length
    #####################################################################################################################
    #Get and item from data
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]
    #####################################################################################################################
    #Get the batch of data
    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]
    #####################################################################################################################
    #Append the data, if full then remove the first element and append the new data in its place
    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, set the index to appen as the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        # Append the data 
        self.data[(self.start + self.length - 1) % self.maxlen] = v

# This ensures that the shape is good with the observation/reward/action
def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    '''
    This class uses the ring buffer object to create the dynamic memory. and has the necessary funcitons to sample batch data
    '''
    def __init__(self, limit, action_shape, observation_shape,additionalObservation_shape):
        self.limit = limit
        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.observations0Additional = RingBuffer(limit,shape=additionalObservation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        #self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)
        self.observations1Additional = RingBuffer(limit,shape=additionalObservation_shape)
        if gv.UseSoftDataUpdating:
            self.observationsSoft = RingBuffer(limit,shape=observation_shape)
        #self.obsrvation00=RingBuffer(limit,shape=observation_shape)
    #####################################################################################################################
    #Sample a batch of train for training and return it in dictionary format
    def sample(self, batch_size):
        # Draw such that we always have a proceeding element. The self.nb_entries - 2 ensures
        # that we can sample always obs0, obs1, action, reward
        batch_idxs = np.random.randint(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        #terminal1_batch = self.terminals1.get_batch(batch_idxs)
        obs0_additional_batch = self.observations0Additional.get_batch(batch_idxs)
        obs1_additional_batch = self.observations1Additional.get_batch(batch_idxs)
        if gv.UseSoftDataUpdating:
            obs_soft_batch = self.observationsSoft.get_batch(batch_idxs)
        #obs00_batch = self.obsrvation00.get_batch(batch_idxs)
        if gv.UseSoftDataUpdating:
            result = {
                'obs0': array_min2d(obs0_batch),
                'obs1': array_min2d(obs1_batch),
                'rewards': array_min2d(reward_batch),
                'actions': array_min2d(action_batch),
                'obs0_additional': array_min2d(obs0_additional_batch),
                'obs1_additional': array_min2d(obs1_additional_batch),
                'obs_soft': array_min2d(obs_soft_batch)
            }
        else:
            result = {
                'obs0': array_min2d(obs0_batch),
                'obs1': array_min2d(obs1_batch),
                'rewards': array_min2d(reward_batch),
                'actions': array_min2d(action_batch),
                'obs0_additional': array_min2d(obs0_additional_batch),
                'obs1_additional': array_min2d(obs1_additional_batch),
            }
        return result
    #####################################################################################################################
    #Append the data into the Ring buffers
    if gv.UseSoftDataUpdating:
        def append(self,obs0,obs0_additional, action, obs_soft,reward, obs1,obs1_additional, training=True):
            if not training:
                return       
            self.observations0.append(obs0)
            self.actions.append(action)
            self.rewards.append(reward)
            self.observations1.append(obs1)
            self.observations0Additional.append(obs0_additional)
            self.observations1Additional.append(obs1_additional)
            self.observationsSoft.append(obs_soft)
    else:
        def append(self,obs0,obs0_additional, action, reward, obs1,obs1_additional, training=True):
            if not training:
                return        
            self.observations0.append(obs0)
            self.actions.append(action)
            self.rewards.append(reward)
            self.observations1.append(obs1)
            self.observations0Additional.append(obs0_additional)
            self.observations1Additional.append(obs1_additional)

    @property
    def nb_entries(self):
        return len(self.observations0)

