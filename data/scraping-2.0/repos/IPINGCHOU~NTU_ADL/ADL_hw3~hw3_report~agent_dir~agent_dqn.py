import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from agent_dir.agent import Agent
from environment import Environment
from collections import namedtuple

use_cuda = torch.cuda.is_available()

# h-para
DEVICE = 'cuda:0'
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
# MEMORY_CAPACITY = 10000
DQN_TYPE = 'double dueling DQN'
TARGET_UPDATE_FREQ = 1000
MAX_TIMESTEPS = 200000
BATCH_SIZE = 1024

class DuelingDQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.feature = nn.Sequential(
            nn.Linear(3136,512),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            # nn.Linear(512, 512),
            # nn.LeakyReLU(0.01),
            nn.Linear(512, num_actions)
        )

        self.value = nn.Sequential(
            # nn.Linear(512, 512),
            # nn.LeakyReLU(0.01),
            nn.Linear(512, 1)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.feature(x)
        adv = self.advantage(x)
        v = self.value(x)
        return v + adv - adv.mean()


# class DuelingDQN(nn.Module):
#     '''
#     This architecture is the one from OpenAI Baseline, with small modification.
#     '''
#     def __init__(self, channels, num_actions):
#         super(DuelingDQN, self).__init__()
#         self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.fc = nn.Linear(3136, 512)
#         self.head = nn.Linear(512, num_actions)

#         self.relu = nn.ReLU()
#         self.lrelu = nn.LeakyReLU(0.01)

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.lrelu(self.fc(x.view(x.size(0), -1)))
#         q = self.head(x)
#         return q


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        # save a transtion
        if len(self.memory) < self.capacity: # if still has capacity, init memory[position]
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args) # push
        self.position = (self.position + 1) % self.capacity # make position stays in [0, capacity]
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n

        # build target, online network
        self.target_net = DuelingDQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.to(DEVICE) if use_cuda else self.target_net
        self.online_net = DuelingDQN(self.input_channels, self.num_actions)
        self.online_net = self.online_net.to(DEVICE) if use_cuda else self.online_net

        if args.test_dqn:
            self.load('dqn')

        # discounted reward
        self.GAMMA = 0.99

        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = BATCH_SIZE
        self.num_timesteps = MAX_TIMESTEPS # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.target_update_freq = TARGET_UPDATE_FREQ # frequency to update target network
        self.buffer_size = 10000 # max size of replay buffer

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0 # num. of passed steps

        # TODO: initialize your replay buffer
        self.memory = ReplayMemory(self.buffer_size)

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location = DEVICE))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location = DEVICE))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass

    def make_action(self, state, test=False):
        # TODO:
        # Implement epsilon-greedy to decide whether you want to randomly select
        # an action or not.
        # HINT: You may need to use and self.steps
        sample = random.random()  # get random number from [0.0, 1.0)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps / EPS_DECAY)
        
        if test == True:
            eps_threshold = 0
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).to(DEVICE)

        if sample > eps_threshold:
            with torch.no_grad():
                # if True, return the move with highest reward
                action = self.online_net(state).max(1)[1].view(1,1)

        else:
            # if False, randomly select a move
            action = torch.tensor([[random.randrange(self.num_actions)]], dtype = torch.long).to(DEVICE)

        if test == True:
            return action.item()
        else:
            return action


    def update(self):
        # TODO:
        # step 1: Sample some stored experiences as training examples.
        # step 2: Compute Q(s_t, a) with your model.
        # step 3: Compute Q(s_{t+1}, a) with target model.
        # step 4: Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # step 5: Compute temporal difference loss
        # HINT:
        # 1. You should not backprop to the target model in step 3 (Use torch.no_grad)
        # 2. You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it
        #    is the terminal state.
        if len(self.memory) < self.batch_size:
            return # return if not enough memory to update
        trans = self.memory.sample(self.batch_size)
        batch = self.memory.transition(*zip(*trans))

        # non-final mask, concatenate batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s : s is not None, batch.next_state)), device = DEVICE, dtype = torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # compute Q(s_t, act)
        # Q(s_t) -> model -> output, select columns of actions from output
        state_action_values = self.online_net(state_batch).gather(1, action_batch)
        
        # compute V(s_{t+1}) for all next states
        # get expected values of actions based on the older target_net
        # select their best rewards with man(1)[0]
        # state value or 0 in case the state was final
        next_state_values = torch.zeros(self.batch_size, device = DEVICE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # compute huber loss
        # print(state_action_values.shape)
        # print(expected_state_action_values.unsqueeze(1).shape)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # update the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()

        return loss.item()

    def train(self):
        print('Now using ' + DQN_TYPE + ' model' )
        print('Now setting target update freq: ' + str(TARGET_UPDATE_FREQ))
        record_reward = []
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        loss = 0
        best_avg = 0
        single_episode_reward = 0
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).to(DEVICE)

            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state).to(DEVICE)
                next_state, reward, done, _ = self.env.step(action.item())
                total_reward += reward
                single_episode_reward += reward
                reward = torch.tensor([reward]).to(DEVICE)

                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.to(DEVICE)
                # reward = reward.to(DEVICE)

                # TODO: store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # TODO: update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                # if (self.steps % self.save_freq == 0):
                #     self.save('dqn')

                self.steps += 1
            record_reward.append(single_episode_reward)
            # print(single_episode_reward)
            single_episode_reward = 0

            if episodes_done_num % self.display_freq == 0:
                avg_reward = total_reward / self.display_freq
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, avg_reward, loss))
                total_reward = 0
                # if avg_reward > best_avg and avg_reward >= 30:
                #     self.save('DuelingDQN')
                #     best_avg = avg_reward
                np.save('small_dueling_dqn_reward_' + str(TARGET_UPDATE_FREQ), np.array(record_reward))

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                # self.save('dqn_final')
                break
