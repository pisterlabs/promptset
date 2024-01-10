# The code is based on the ACKTR implementation from OpenAI baselines
# This code implements method by Kangin & Pugeault "On-Policy Trust Region Policy Optimisation with Replay Buffers"
import numpy as np

class ReplayBufferParameters:
    def __init__ (self, replay_buffer_size, sample_size=-1):
        self.replay_buffer_size = replay_buffer_size
        if sample_size > 0:
            self.sample_size = sample_size
        else:
            self.sample_size = self.replay_buffer_size

class ReplayBuffer:
    def __init__ (self, parameters):
        self.buff = []
        self.parameters = parameters

    def clear (self):
        self.buff = []

    def push (self, value):
        self.buff.append (value)
        if len(self.buff) > self.parameters.replay_buffer_size:
             self.buff = self.buff [1:]

    def get (self):
       return self.buff

    def sample (self):
       indices = np.random.permutation (len(self.buff))
       
       if self.parameters.sample_size < len(indices): 
            indices = indices[:self.parameters.sample_size]
       #indices.reshape ((1, -1))
       #print (indices)
       return [self.buff[i] for i in indices]
