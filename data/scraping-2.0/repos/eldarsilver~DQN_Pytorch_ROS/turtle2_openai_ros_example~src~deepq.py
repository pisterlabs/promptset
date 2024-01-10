#!/usr/bin/env python

import gym
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from functools import reduce
import pickle



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import math
import glob
import io
import base64
from memory import ReplayMemory
import datetime
import json




class DQN(nn.Module):
    # hidden_size=64
    def __init__(self, inputs, outputs, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=inputs, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=outputs)
        #self.fc5 = nn.Linear(in_features=16, out_features=outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc4(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        #x = self.fc5(x)
        return x


def compute_eps_threshold(step, eps_start, eps_end, eps_decay):
    # eps_start = 1.0, eps_end = 0.1, eps_decay = num_steps = 9e4
    # 0.1 + 0.9 * math.exp(-1. * step / 9e4)
    # with step = 9e4: 0.1 + 0.9 * 0.36787944117 = 0.1 + 0.33109149705 = 0.43109149705
    # with step = 0: 0.1 + 0.9 * math.exp(-1 * 0) = 0.1 + 0.9 * 1 = 1.0
    #return eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)
    return eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)


def select_action(policy, state, device, env, eps_greedy_threshold, n_actions):
    rospy.logwarn("eps_greedy_threshold: " + str(eps_greedy_threshold))
    if random.random() > eps_greedy_threshold:
        rospy.logwarn("Entering select action random.random() > eps_greedy_threshold...")
        policy_used = True
        #rospy.logwarn("state.shape: ")
        #rospy.logwarn(state.shape)
        #rospy.logwarn("n_actions Env.action_space.n%d", n_actions)
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policy.eval()
            action = policy(state).max(1)[1].view(1, 1)
            policy_act = action
            policy.train()
    else:
        rospy.logwarn("Entering select action random.random() < eps_greedy_threshold...")
        policy_used = False
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        with torch.no_grad():
            policy.eval()
            policy_act = policy(state).max(axis=1)[1].view(1, 1)
    return action, policy_act, policy_used


def train(policy_net, target_net, optimizer, scheduler, memory, batch_size, gamma, device, env):
    if len(memory) < batch_size:
        return
    full_memory = memory.sample(len(memory))
    full_memory_fields = memory.Transition(*zip(*full_memory))
    full_rewards = torch.cat(full_memory_fields.reward).float()
    #full_states = torch.cat(full_memory_fields.state)
    transitions = memory.sample(batch_size)
    # This converts batch-array of Transitions to Transition of batch-arrays.
    # list of Transitions: [(s, a, r, s', d), (s, a, r, s', d), ...]
    # will become: 
    # Transition((s0, s1, s2, ...), (a0, a1, a2, ...), ...)
    batch = memory.Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    e = np.finfo(np.float32).eps.item()
    state_batch = torch.cat(batch.state)
    #state_batch = (state_batch - full_states.mean()) / (full_states.std() + e)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward) #.float()
    #reward_batch = (reward_batch - full_rewards.mean()) / (full_rewards.std() + e)

    # Compute Q(s_t, a) - the model computes Q(s_t) for all a, then we select the columns of actions taken.
    #rospy.logwarn("state_batch.shape: ")
    #rospy.logwarn(state_batch.shape)
    #rospy.logwarn("n_inputs Env.observation_space: ")
    #rospy.logwarn(env.observation_space)
    #rospy.logwarn("n_inputs Env.observation_space.shape: ")
    #rospy.logwarn(env.observation_space.shape)
    #rospy.logwarn("n_inputs Env.observation_space.shape[0] %d", n_inputs)
    #rospy.logwarn("n_actions Env.action_space.n %d", n_actions)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    #rospy.logwarn("state_action_values.shape: ")
    #rospy.logwarn(state_action_values.shape)
    #rospy.logwarn("state_action_values: ")
    #rospy.logwarn(state_action_values)

    # Compute Q(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # Note the call to detach() on Q(s_{t+1}), which prevents gradient flow
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(dim=1)[0].detach() 
    # Compute targets for Q values: y_t = r_t + gamma * max(Q_{t+1})
    expected_state_action_values = reward_batch + (gamma * next_state_values)
    """
    rospy.logwarn("expected_state_action_values.shape: ")
    rospy.logwarn(expected_state_action_values.shape)
    rospy.logwarn("expected_state_action_values: ")
    rospy.logwarn(expected_state_action_values)
    rospy.logwarn("expected_state_action_values.unsqueeze(1): ")
    rospy.logwarn(expected_state_action_values.unsqueeze(1))
    """
    # Compute Pseudo-Huber loss between predicted Q values and targets y
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # Take an SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    for name, weight in policy_net.named_parameters():
        """
        print("\nname: ")
        print(name)
        print("\nweight: ")
        print(weight)
        print("\nweight.grad: ")
        print(weight.grad)
        """
        writer.add_histogram(name, weight, step_count)
        writer.add_histogram(str(name) + '/grad', weight.grad, step_count)


def test(env, policy_net, device, test_global_step, render=False):
    state, ep_reward, done = env.reset(), 0, False
    rospy.logwarn("Entering test method ...")
    test_local_step = 0
    while not done:
        if render:
            env.render()
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        action, policy_act, policy_used  = select_action(policy_net, state, device, env, eps_greedy_threshold=0., n_actions=1)
        state, reward, done, _ = env.step(action.item())
        test_local_step += 1
        test_global_step += 1
        rospy.logwarn("Testing: Reward of this step: ")
        rospy.logwarn(reward)
        #writer.add_scalar("Test_step_Reward", reward, global_step=test_local_step)
        ep_reward += reward
        rospy.logwarn("Testing: Cumulative Reward of this episode: ")
        rospy.logwarn(ep_reward)
        writer.add_scalar("Test_Cumulative_Rewards", ep_reward, global_step=test_global_step)
    return ep_reward, test_global_step



if __name__ == '__main__':

    rospy.init_node('example_turtlebot2_maze_dqn', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param('/turtlebot2/task_and_robot_environment_name')
    # Create the Gym environment
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")


    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('turtle2_openai_ros_example')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    """
    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/turtlebot2/alpha")
    Epsilon = rospy.get_param("/turtlebot2/epsilon")
    Gamma = rospy.get_param("/turtlebot2/gamma")
    epsilon_discount = rospy.get_param("/turtlebot2/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot2/nepisodes")
    nsteps = rospy.get_param("/turtlebot2/nsteps")
    running_step = rospy.get_param("/turtlebot2/running_step")
    """

    # Hyperparameters
    gamma = 0.999  # initially 0.99 discount factor
    seed = 543  # random seed
    log_interval = 25  # controls how often we log progress, in episodes
    num_steps = 15e4  # 11e4 number of steps to train on
    batch_size = 512  # batch size for optimization
    lr = 1e-3  # 1e-4learning rate
    eps_start = 1.0  # initial value for epsilon (in epsilon-greedy)
    eps_end = 0.1  # final value for epsilon (in epsilon-greedy)
    eps_decay = 9e4  # 8e4 num_steps, length of epsilon decay, in env steps
    target_update = 1000  # how often to update target net, in env steps
    test_global_step = 0 # Global number of testing steps for tracking cummulative rewards in Tensorboard

    # If gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Fix random seed (for reproducibility)
    env.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Get number of actions from gym action space
    #n_inputs = env.observation_space.shape[0] # 76
    n_inputs = 5 # 76, 720, 360
    n_actions = env.action_space.n
    #rospy.logwarn("n_inputs Env.observation_space.shape[0] %d", n_inputs)
    #rospy.logwarn("n_actions Env.action_space.n %d", n_actions)

    
 

    policy_net = DQN(n_inputs, n_actions).to(device)
    target_net = DQN(n_inputs, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    # Decays the learning rate of each parameter group by gamma every step_size epochs. Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler. When last_epoch=-1, sets initial lr as lr.
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma=0.9)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=int(num_steps))
    memory = ReplayMemory(10000)


    ############################################################################
    #logdir = os.path.join("$HOME/python3_ws/src/turtle2_openai_ros_example/src/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    basedir = os.path.dirname(__file__)
    basedirpathlogs = os.path.join(basedir, "logs")
    logdir = os.path.join(basedirpathlogs, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=logdir)
    #tracedir = "$HOME/python3_ws/src/turtle2_openai_ros_example/src/trace"
    tracedir = os.path.join(basedir, "trace")
    ############################################################################

    print("Target reward: {}".format(env.spec.reward_threshold))
    step_count = 0
    ep_rew_history = []
    i_episode, ep_reward = 0, -float('inf')
    while step_count < num_steps:
        rospy.logdebug("############### START EPISODE=>" + str(i_episode))
        # Initialize the environment and state
        # type(state): <class 'list'>
        state, done = env.reset(), False
        state = [round(num, 1) for num in state]
        list_state = state
        #print("\n type(state): ")
        #print(type(state))
        rospy.logwarn("# state we are => " + str(state))
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        while not done:
            rospy.logwarn("i_episode: " + str(i_episode))
            rospy.logwarn("step_count: " + str(step_count))
            # Select an action
            eps_greedy_threshold = compute_eps_threshold(step_count, eps_start, eps_end, eps_decay)
            action, policy_act, policy_used = select_action(policy_net, state, device, env, eps_greedy_threshold, n_actions)
            rospy.logwarn("Next action is:%d", action)

            # Perform action in env
            next_state, reward, done, _ = env.step(action.item())
            next_state = [round(num, 1) for num in next_state]
            list_next_state = next_state
            #rospy.logwarn(str(next_state) + " " + str(reward))

            # Bookkeeping
            next_state = torch.from_numpy(np.array(next_state)).float().unsqueeze(0).to(device)
            #reward = reward_shaper(reward, done)
            reward = torch.tensor([reward], device=device)
            step_count += 1
            

            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            memory.push_trace(i_episode, step_count, list_state, action.item(), list_next_state, reward.item(), policy_act.item(), eps_greedy_threshold, policy_used)

            """
            # Make the algorithm learn based on the results
            rospy.logwarn("# state we were=>" + str(state))
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            #rospy.logwarn("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logwarn("# State in which we will start next step=>" + str(next_state))
            """

            # Move to the next state
            state = next_state
            list_state = list_next_state

            # Perform one step of the optimization (on the policy network)
            train(policy_net, target_net, optimizer, scheduler, memory, batch_size, gamma, device, env)
            """
            for name, weight in policy_net.named_parameters():
                writer.add_histogram(name, weight, step_count)
                writer.add_histogram(str(name) + '/grad', weight.grad, step_count)
            """
          
            # Update the target network, copying all weights and biases in DQN
            if step_count % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                #############################################################################################################
                #torch.save(policy_net.state_dict(), '$HOME/python3_ws/src/turtle2_openai_ros_example/src/checkpoints/dqn-episode-{0}-step-{1}.pt'.format(str(i_episode), str(step_count)))
                model_dir = os.path.dirname(__file__)
                MODEL_PATH = os.path.join(model_dir, 'checkpoints/dqn-episode-{0}-step-{1}.pt'.format(str(i_episode), str(step_count)))
                torch.save(policy_net.state_dict(), MODEL_PATH)
                #torch.save(policy_net.state_dict(), 'checkpoints/dqn-episode-{0}-step-{1}.pt'.format(str(i_episode), str(step_count)))
                fname = datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S") + ".json"
                list_namedtuple = memory.get_memtrace()
                with open(os.path.join(tracedir, fname), 'w') as f:
                    json.dump([elem._asdict() for elem in list_namedtuple[-1000:-1]], f)
                #torch.save(policy_net.state_dict(), '/home/eldar/python3_ws/src/turtle2_openai_ros_example/src/checkpoints/dqn-{}.pt'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        i_episode += 1

        for name, weight in policy_net.named_parameters():
            """
            print("\nname: ")
            print(name)
            print("\nweight: ")
            print(weight)
            print("\nweight.grad: ")
            print(weight.grad)
            """
            writer.add_histogram(name, weight, step_count)
            #writer.add_histogram('grad', weight.grad, step_count)
        


        # Evaluate greedy policy
        if i_episode % log_interval == 0 or step_count >= num_steps:
            ep_reward, test_global_step = test(env, policy_net, device, test_global_step)
            ep_rew_history.append((i_episode, ep_reward))
            print('Episode {}\tSteps: {:.2f}k'
                  '\tEval reward: {:.2f}'.format(
                  i_episode, step_count/1000., ep_reward))
    
    print("\nFinished training! Eval reward: {:.2f}".format(ep_reward))
    print("\nFinished training! List of Eval rewards: ")
    print(ep_rew_history)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    str_i_episode = str(i_episode)
    #######################################################################################
    #torch.save(policy_net.state_dict(), '$HOME/python3_ws/src/turtle2_openai_ros_example/src/checkpoints/dqn-final-episode-{0}-step-{1}.pt'.format(str_i_episode, str(step_count)))
    model_dir = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(model_dir, 'checkpoints/dqn-episode-{0}-step-{1}.pt'.format(str(i_episode), str(step_count)))
    torch.save(policy_net.state_dict(), MODEL_PATH)
    



    
