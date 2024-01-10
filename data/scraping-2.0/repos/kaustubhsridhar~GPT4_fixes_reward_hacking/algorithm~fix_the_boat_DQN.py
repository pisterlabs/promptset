# DQN implementation using PyTorch for the boat-grid environment
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from envs.boat_grid import BoatRaceEnvironment
from copy import deepcopy
import requests
import json
import openai
import re
from io import StringIO
import sys

from agents.DQNmodel import DQN
from viz.visualization import *

import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
openai.api_key = # Put your OpenAI API key here within ""

# Define the DQN Agent

class DQNAgent:
    def __init__(self, env, state_dim, action_dim, lr=0.0001, gamma=0.999, epsilon_max=0.99, epsilon_min=0.01,\
                 epsilon_decay=0.99, decay_count=200000, tau = 0.01, batch_size=128, memory_size=10000, K=9):
        self.llm = 'gpt-4'
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_method = 'linear'
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.K = K
        self.headers = {
            "content-type": "application/x-www-form-urlencoded",
            "X-RapidAPI-Key": "82635cda44msh93bc6645248df15p1bd56bjsnd5fb18e94197",
            "X-RapidAPI-Host": "chatgpt-4-bing-ai-chat-api.p.rapidapi.com"
        }
        self.url = "https://chatgpt-4-bing-ai-chat-api.p.rapidapi.com/chatgpt-4-bing-ai-chat-api/0.2/send-message/"
        self.bing_u = "1NkCBd2rXBcDWPTAaArBzjWhcS-j5__uE4hZlnsHE3dH2ODJDzZ56ctX-y4F6qILm2L9xZOXsODCcJ5fxhPkgaN4-9gajpY8nqao03WFZSvff3Vqinr4O-nZNq_1k5xspvYZMtk0IgU24OTeQp5jISefDuwphgga8LGeY-58wIogkvF5AsdqP5HRi-IhmRn64t5hS8fDDRhqipgqQtetymw" # "1Nk8BrZXcrspIizGK7OZPKjnO8zvSwB0HPVq7Njmwir8Xfcb1EynX8MbN8IehzpswnFupl9pdBWgz99j3tnQQywpyKJq1SoW67hK3SBrChheYNZVTO2tpH6mPI3Z1eSsYeFQvSPvdUvk6PS4AdM0Q_xb_pQZeHLGVNaAQa--WIn1TRInNLMPmgA4IfQ_jlRBNQmc6xX-k11uoKT0oPutK5cSqreILjtwkHctI9X4DKZ0"
        with open(f'assets/init_prompt.txt', 'r') as f:
            self.initial_prompt = f.read()
        
        with open(f'assets/final_prompt.txt', 'r') as f:
            self.final_prompt = f.read()

        assert K < 100 # num steps in a episode
        self.memory = deque(maxlen=memory_size)
        self.window = deque(maxlen=K)
        self.last_few_states = deque(maxlen=4)
        self.tau = tau
        self.decay_count = decay_count
        self.base_reward_thresh = 20

        self.model = DQN(state_dim, action_dim)
        self.model = self.model.to(device)

        self.target_model = DQN(state_dim, action_dim)
        self.target_model = self.target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.update_count = 0

        self.function_string = ''
        self.reward_modified = False        
        
    def convert_string(self, input_string):
        input_string = input_string.replace('\n', ' ')
        input_string = input_string.replace('\t', '')
        input_string = input_string.replace(' ', '%20')
        input_string = input_string.replace(':', '%3A')
        input_string = input_string.replace(',', '%2C')
        input_string = input_string.replace('[', '%5B')
        input_string = input_string.replace(']', '%5D')
        input_string = input_string.replace('\n', '%0A')
        output_string = input_string.replace("=", '%3D')
        return output_string

    def parse(self, input_string):        
        # Extract the Python function between <START> and <END>, meant to be the new rewar
        pattern = re.compile(r'<START>(.*?)<END>', re.DOTALL)
        match = pattern.search(input_string)

        if match:
            self.function_string = match.group(1)
            print("Extracted function:\n", self.function_string)
        else:
            print("No function found")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.model(torch.Tensor(state).to(device))
                return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.window.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.Tensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.Tensor(rewards).to(device)
        next_states = torch.Tensor(next_states).to(device)
        dones = torch.Tensor(dones).to(device)
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        max_next_q_values = self.target_model(next_states).detach().max(1)[0]

        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        
        if self.update_count % 10 == 0:
            self.target_train()


    def target_train(self):
        # Exponential moving average
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update_epsilon(self):
        if self.decay_method == 'linear':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        elif self.decay_method == 'exponential':
            self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-1. * self.update_count / self.decay_count)        

    def check_for_reward_hacking(self):
        # Convert the states to strings
        states = np.array(self.window, dtype=object)[:,0]
        states = (states + 1.0)*1.5 # De-normalizing the states
        str_states = ''
        for i in range(len(states)):
            str_states += np.array2string(states[i].reshape(5,5)) + '\n\n'
        
        states_reshaped = np.array([s.reshape((5,5)) for s in states])
        create_gif(states_reshaped, "hacked")
        full_prompt = self.initial_prompt + str_states + self.final_prompt

        if self.llm == 'gpt-4':
            messages = [{"role": "user", "content": full_prompt}]
            response = openai.ChatCompletion.create(
                model=self.llm,
                messages=messages,
                temperature=0.5, # temperature 0 keeps it deterministic
            )
            text_output = response['choices'][0]['message']['content']
        elif self.llm == 'bing':
            total_payload = "bing_u_cookie=" + self.bing_u + "&question=" + self.convert_string(full_prompt)
            response = requests.request("POST", self.url, data=total_payload, headers=self.headers) # Calling Bing Chat API
            response_text = response.text
            json_data = json.loads(response_text)
            text_output = json_data[0]["text_output"]

        print(text_output, f'\n')
        is_hacked = False
        if "[TRUE]" in text_output:
            is_hacked = True
            print("\nReward Hacking Detected!\n")
        
        self.parse(text_output)

        return is_hacked
    
    def train(self, epochs=1000):
        # Train the agent
        total_reward = 0
        is_hacked = False
        reward_list = deque(maxlen=5)
        for episode in range(epochs):
            timestep = self.env.reset()
            state = timestep.observation['board'] / 1.5 - 1.0
            state = state.reshape(-1)
            self.last_few_states.append(state.reshape(5, 5))
            done = False
            num_steps = 0
            total_previous_reward = total_reward
            total_reward = 0
            while not done:
                num_steps += 1
                self.update_count += 1
                action = self.act(state)
                timestep = self.env.step(action)
                next_state = timestep.observation['board']
                next_state = next_state.reshape(-1) / 1.5 - 1.0
                self.last_few_states.append(next_state.reshape(5, 5))
                reward = timestep.reward
                if not is_hacked:
                    reward = 0.75 if reward == 2.0 else -0.25
                if is_hacked:
                    if num_steps >= 3:
                        last_few_states = np.array(self.last_few_states)
                        last_few_states = (last_few_states + 1.0)*1.5 # De-normalizing the states
                        last_few_states = last_few_states.astype(int)
                        exec(self.function_string, globals())
                        reward = reward_function(last_few_states) # reward_function(last_few_states)
                done = timestep.last()
                
                self.remember(deepcopy(state), deepcopy(action), \
                              deepcopy(reward), deepcopy(next_state), \
                                deepcopy(done))

                state = next_state
                total_reward += reward
            
            for _ in range(10):
                self.replay()
            
            self.update_epsilon()
            print(f"Episode {episode} with {self.epsilon=}, {self.update_count=} finished with reward {total_reward}")

            reward_list.append(total_reward)
            if (self.epsilon == self.epsilon_min) and not self.reward_modified and np.mean(np.array(reward_list)) > self.base_reward_thresh:
                print(f'\n')
                is_hacked = self.check_for_reward_hacking()
                if is_hacked:
                    self.reward_modified = True
                    torch.save(self.model.state_dict(), "weights/hacked_DQN.pth")
                    # Save GIF plot from testing
                    
                    # reset variables
                    self.memory = deque(maxlen=self.memory_size)
                    print("Memory after wiping: ", self.memory)
                    self.last_few_states = deque(maxlen=4)
                    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
                    self.epsilon = self.epsilon_max
                    self.epsilon_decay = 0.9

        # Save the model in the weights directory
        torch.save(self.model.state_dict(), "weights/fixed_DQN.pth")
        # Save GIF plot from testing after fixing the reward function
        create_gif(self.test(), "fixed")

    def test(self):
        # Test the agent
        self.model.load_state_dict(torch.load("weights/DQN.pth"))
        timestep = self.env.reset()
        state = timestep.observation['board']
        all_states = [deepcopy(state)]
        print(state)
        state = state.reshape(-1) / 1.5 - 1.0
        done = False
        total_reward = 0
        while not done:
            action = self.act(state)
            timestep = self.env.step(action)
            next_state = timestep.observation['board']
            all_states.append(deepcopy(next_state))
            print(next_state)
            next_state = next_state.reshape(-1) / 1.5 - 1.0
            reward = timestep.reward
            reward = 0.75 if reward == 2.0 else -0.25
            total_reward += reward
            done = timestep.last()
            state = next_state

        return np.array(all_states).astype(int)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='seed for random number generator')
    args = parser.parse_args()

    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Define the environment
    env = BoatRaceEnvironment()
    agent = DQNAgent(env, state_dim=25, action_dim=4)

    # Train the agent
    agent.train(epochs=800)
