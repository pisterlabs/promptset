import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from collections import namedtuple
import matplotlib.pyplot as plt
from agent_dir.agent import Agent
from environment import Environment
import pickle
use_cuda = torch.cuda.is_available()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

#experience replay
#break correlations in data, bring us back to iid setting
#Learn from all past policies
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory=[]
        self.position= 0
    def push(self, *args):
        '''
        Save a transition
        '''
        if len(self.memory) < self.capacity:
            self.memory.append(None) #append a None or index will go out of range
        #print(*args)
        #https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/
        #input()
        self.memory[self.position]= Transition(*args) 
        self.position= (self.position+1)% self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):  #https://blog.csdn.net/u013061183/article/details/74773196
        return len(self.memory)


class DQN(nn.Module): #this is the critic, the value function; Q-networks represent value functions with weights w 
    ''' 
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)
#DQN will compute the expected return of taking each action(total 7 actions) given the current state

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q

class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4 #what is input channels ??? -->take the last 4 frames
        self.num_actions = self.env.action_space.n # 7 actions 
        # TODO:
        # Initialize your replay buffer
        self.memory = ReplayMemory(10000)
        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, self.num_actions)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load('dqn')
        
        # discounted reward
        self.GAMMA = 0.99
        
        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 300000 # total training steps -->you can change it to 1000000 in report Q2
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network

        #epsilon greedy policy hyperparameters
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay= 200
        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)
        #fix the target net, update the parameters in the online_net
        #freeze target Q network to avoid oscillation
        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration
        self.device= torch.device("cuda" if use_cuda else "cpu")

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass
    
    def make_action(self, state, test=False):
        # TODO:
        if not test:
        # At first, you decide whether you want to explore the environemnt
            sample= random.random()
            #implementation of epsilon greedy exploration algorithms
            eps_threshold = self.eps_end +(self.eps_start-self.eps_end)*math.exp(-1*self.steps/self.eps_decay)
        #self.steps +=1 #see def train(), already in there
        # TODO:
        # if explore, you randomly samples one action
        # else, use your model to predict action
            if sample >  eps_threshold: #no explore
                with torch.no_grad():
                    #print(self.online_net(state))
                    #print(self.online_net(state).shape) #torch.Size([1, 7])
                    #print(self.online_net(state).max(1)) #(tensor([0.0511], device='cuda:0'), tensor([6], device='cuda:0'))
                    #print(self.online_net(state).max(1)[1].view(1, 1)) #tensor([[4]], device='cuda:0')
                    action= self.online_net(state).max(1)[1].view(1, 1) #policy function is right here
                    #pi'(s) = argmax(a) Q(s, a)
                    #the policy function does not have extra parameters, it depends on the value function
                    #not suitable for continuous action
            else: #if explore
                #about exploration:https://www.youtube.com/watch?v=3htlYl4Whr8&t=2006s 55:18
                action=torch.tensor([[random.randrange(self.num_actions)]], device= self.device, dtype= torch.long)
        else:
            #print(state)
            #print(state.shape) #(84, 84, 4)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            #(84, 84, 4) -->torch.size([1 ,4, 84, 84])
            state = state.cuda() if use_cuda else state
            action= self.online_net(state).max(1)[1].item()
            #print(action)
            

        return action

    def update(self): 
        # TODO:
        # To update model, we sample some stored experiences as training examples.
        if len(self.memory) < self.batch_size:
            return
        transitions= self.memory.sample(self.batch_size)
        #print(self.num_actions)
        #input()
        #print(len(transitions)) #a list of transitions, len = batch_size
        #input()
        # TODO:
        # Compute Q(s_t, a) with your model.
        #print(zip(*transitions)) #<zip object at 0x7f5d6b1b4788>
        #print(*zip(*transitions)) #dereference the zip object
        batch= Transition(*zip(*transitions))
        non_final_mask =torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8)
        #if we are at the final state, next_state will be None!!! So be careful.
        #print(non_final_mask) #tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.uint8)
        #python map function: http://www.runoob.com/python/python-func-map.html
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        #print(len(batch.state)) #batch.state is a tuple #batch_size
        state_batch = torch.cat(batch.state)
        #print(state_batch.shape) # batch_size, 4, 84, 84
        
        action_batch = torch.cat(batch.action)
        #print(action_batch.shape) #batch_size, 1
        #print(action_batch)
        reward_batch = torch.cat(batch.reward)
        #print(reward_batch.shape) #batch_size
        #print(batch)
        #print(self.online_net(state_batch).shape) #batch_size, num_of_action
        state_action_values = self.online_net(state_batch).gather(1, action_batch)
        #
        #print(state_action_values.shape) #torch.Size([32, 1])
        #input()
        with torch.no_grad():
            #print("HAHA")
            # TODO:
            # Compute Q(s_{t+1}, a) for all next states.
            # Since we do not want to backprop through the expected action values,
            # use torch.no_grad() to stop the gradient from Q(s_{t+1}, a)
            next_state_values= torch.zeros(self.batch_size, device= self.device)
            #print(self.target_net(non_final_next_states))
            #print(self.target_net(non_final_next_states).shape) #torch.Size([32, 7]) #batch_size, num_of_actions 
            #each sample has 7 values corresponding to 7 actions, given the next_state
            #print(self.target_net(non_final_next_states).max(1))
            #print(self.target_net(non_final_next_states).max(1)[0].shape) #torch.Size([32])
            #print(self.target_net(non_final_next_states).max(1)[0].detach()) #Returns a new Tensor, detached from the current graph.
            #If keepdim is True, the output tensors are of the same size as input except in the dimension dim where they are of size 1. Otherwise, dim is squeezed (see torch.squeeze()), resulting in the output tensors having 1 fewer dimension than input.
            #print(next_state_actions)
            #print(next_state_actions.shape) #torch.Size([32, 1])
            #print(self.target_net(non_final_next_states).gather(1, next_state_actions).shape)  #torch.Size([32, 1])
            next_state_actions =self.online_net(non_final_next_states).max(1)[1].unsqueeze(1) #argmax(a') Q(s', a', w)
            next_state_values[non_final_mask]= self.target_net(non_final_next_states).gather(1, next_state_actions).squeeze().detach()
            #you must detach() or the performance will drop
            #input()
        # TODO:
        # Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it is the terminal state.
        expected_state_action_values = (self.GAMMA*next_state_values) +reward_batch
        #expected_state_action_values is the learning target!!!
        #value functions decompose into a Bellman equation
        #print(expected_state_action_values)
        #print(expected_state_action_values.shape) #torch.Size([32]) batch_size
        #whole process: https://www.youtube.com/watch?v=3htlYl4Whr8&t=2006s 53:41
        # TODO:
        # Compute temporal difference loss -->HOW????
        #update value toward estimated return
        #https://www.youtube.com/watch?v=3htlYl4Whr8  31:00
        #in pytorch tutorial, they use Huber loss, you can try that later
        #https://www.youtube.com/watch?v=3htlYl4Whr8&t=2006s 41:18 MSE loss
        #mse_loss= nn.MSELoss()
        #loss= mse_loss(state_action_values,expected_state_action_values.unsqueeze(1))
        loss= F.smooth_l1_loss(state_action_values,expected_state_action_values.unsqueeze(1))
        #print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print(loss)
        #input()
        return loss.item()

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        loss = 0 
        x=[]
        y=[]
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            
            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action[0, 0].data.item())
                #print(next_state)
                #print(reward)
                #input()
                
                total_reward += reward
                reward= torch.tensor([reward], device= self.device) #NOT so sure
                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state
                if done:
                    next_state = None

                # TODO:
                # store the transition in memory
                self.memory.push(state, action, next_state, reward)
                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save('dqn')

                self.steps += 1

            
            
            if episodes_done_num % self.display_freq == 0:
                x.append(self.steps)
                y.append(total_reward / self.display_freq)  #avg reward in last 10 episodes
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
                total_reward = 0
                #plt.plot(x, y)
                #plt.xlabel('Timesteps')
                #plt.ylabel('Avg reward in last 10 episodes')
                #plt.show()
                #plt.savefig('dqn_baseline.png')

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        self.save('dqn')
        pickle_out_x= open('ddqn_x_0.99.pkl', 'wb')
        pickle_out_y=open('ddqn_y_0.99.pkl', 'wb')
        pickle.dump(x, pickle_out_x)
        pickle.dump(y, pickle_out_y)
        pickle_out_x.close()
        pickle_out_y.close()
