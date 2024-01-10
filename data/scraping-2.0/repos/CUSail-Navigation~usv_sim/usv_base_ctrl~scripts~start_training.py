#!/usr/bin/env python3

import gym
import matplotlib.pyplot as plt
import numpy
import time
import torch
from torch import nn, optim
from gym import wrappers
from functools import reduce
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from collections import deque
import random
from copy import deepcopy
import os

if __name__ == '__main__':

    rospy.init_node('sailboat_learn', anonymous=True, log_level=rospy.INFO)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/sailboat/training/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('usv_sim')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # path to store results
    result_dir = rospy.get_param('/sailboat/training/path_to_results')

    # Existing actor and critic models to start from
    load_actor = rospy.get_param('/sailboat/training/load_models/load_actor')
    load_critic = rospy.get_param('/sailboat/training/load_models/load_critic')

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    buffer_size = rospy.get_param('/sailboat/training/buffer_size')
    batch_size = rospy.get_param('/sailboat/training/batch_size')
    gamma = rospy.get_param("/sailboat/training/gamma")
    tau = rospy.get_param("/sailboat/training/tau")
    lr_actor = rospy.get_param("/sailboat/training/lr_actor")
    lr_critic = rospy.get_param("/sailboat/training/lr_critic")

    hidden_size1 = rospy.get_param("/sailboat/training/hidden_size1")
    hidden_size2 = rospy.get_param("/sailboat/training/hidden_size2")

    buffer_start = rospy.get_param("/sailboat/training/buffer_start")
    epsilon = rospy.get_param("/sailboat/training/epsilon")
    epsilon_decay = rospy.get_param("/sailboat/training/epsilon_decay")
    eta = rospy.get_param('/sailboat/training/eta')
    eta_decay = rospy.get_param('/sailboat/training/eta_decay')

    print_every = rospy.get_param("/sailboat/training/print_every")

    nepisodes = rospy.get_param("/sailboat/training/nepisodes")

    # Use GPU if possible
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        rospy.loginfo("Using device {}".format(torch.cuda.get_device_name(0)))
    else:
        rospy.logwarn("No GPU detected. Training on CPU is not recommended.")
        device = torch.device("cpu")

    # Deep Deterministic Policy Gradient RL
    class ReplayBuffer():

        def __init__(self, buffer_size):
            self.buffer_size = buffer_size
            self.num_exp = 0
            self.buffer = deque()

        def add(self, s, a, r, t, s2):
            experience = (s, a, r, t, s2)
            if self.num_exp < self.buffer_size:
                self.buffer.append(experience)
                self.num_exp += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)

        def size(self):
            return self.buffer_size

        def count(self):
            return self.num_exp

        def sample(self, batch_size):
            if self.num_exp < batch_size:
                batch = random.sample(self.buffer, self.num_exp)
            else:
                batch = random.sample(self.buffer, batch_size)

            s, a, r, t, s2 = map(numpy.stack, zip(*batch))

            return s, a, r, t, s2

        def clear(self):
            self.buffer = deque()
            self.num_exp = 0

    class NormalizedEnv(gym.ActionWrapper):

        def action(self, action):
            act_k = (self.action_space.high - self.action_space.low) / 2.
            act_b = (self.action_space.high + self.action_space.low) / 2.
            return act_k * action + act_b

        def reverse_action(self, action):
            act_k_inv = 2. / (self.action_space.high - self.action_space.low)
            act_b = (self.action_space.high + self.action_space.low) / 2.
            return act_k_inv * (action - act_b)

    def fanin_(size):
        fan_in = size[0]
        weight = 1. / numpy.sqrt(fan_in)
        return torch.Tensor(size).uniform_(-weight, weight)

    class Critic(nn.Module):

        def __init__(self,
                     state_dim,
                     action_dim,
                     h1=hidden_size1,
                     h2=hidden_size2,
                     init_w=3e-3):
            super(Critic, self).__init__()

            self.linear1 = nn.Linear(state_dim, h1)
            self.linear1.weight.data = fanin_(self.linear1.weight.data.size())

            self.ln1 = nn.LayerNorm(h1)

            self.linear2 = nn.Linear(h1 + action_dim, h2)
            self.linear2.weight.data = fanin_(self.linear2.weight.data.size())

            self.ln2 = nn.LayerNorm(h2)

            self.linear3 = nn.Linear(h2, 1)
            self.linear3.weight.data.uniform_(-init_w, init_w)

            self.relu = nn.ReLU()

        def forward(self, state, action):
            x = self.linear1(state)
            x = self.ln1(x)
            x = self.relu(x)

            x = self.linear2(torch.cat([x, action], 1))
            x = self.ln2(x)
            x = self.relu(x)

            x = self.linear3(x)

            return x

    class Actor(nn.Module):

        def __init__(self,
                     state_dim,
                     action_dim,
                     h1=hidden_size1,
                     h2=hidden_size2,
                     init_w=0.003):
            super(Actor, self).__init__()

            self.linear1 = nn.Linear(state_dim, h1)
            self.linear1.weight.data = fanin_(self.linear1.weight.data.size())

            self.ln1 = nn.LayerNorm(h1)

            self.linear2 = nn.Linear(h1, h2)
            self.linear2.weight.data = fanin_(self.linear2.weight.data.size())

            self.ln2 = nn.LayerNorm(h2)

            self.linear3 = nn.Linear(h2, action_dim)
            self.linear3.weight.data.uniform_(-init_w, init_w)

            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()

        def forward(self, state):
            x = self.linear1(state)
            x = self.ln1(x)
            x = self.relu(x)

            x = self.linear2(x)
            x = self.ln2(x)
            x = self.relu(x)

            x = self.linear3(x)
            x = self.tanh(x)
            return x

        def get_action(self, state):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.forward(state)
            return action.detach().cpu().numpy()[0]

    class OrnsteinUhlenbeckActionNoise():

        def __init__(self, mu=0, sigma=0.2, theta=.15, dt=1e-2, x0=None):
            self.theta = theta
            self.mu = mu
            self.sigma = sigma
            self.dt = dt
            self.x0 = x0
            self.reset()

        def __call__(self):
            x = self.x_prev + self.theta * (
                self.mu - self.x_prev) * self.dt + self.sigma * numpy.sqrt(
                    self.dt) * numpy.random.normal(size=self.mu.shape)

            self.x_prev = x
            return x

        def reset(self):
            self.x_prev = self.x0 if self.x0 is not None else numpy.zeros_like(
                self.mu)

        def __repr__(self):
            return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                self.mu, self.sigma)

    class GaussianActionNoise():

        def __init__(self, mu=0, sigma=0.2):
            self.mu = mu
            self.sigma = sigma

        def __call__(self):
            return numpy.random.normal(self.mu, self.sigma)

        def __repr__(self):
            return 'GaussianActionNoise(mu={}, sigma={})'.format(
                self.mu, self.sigma)

    # training
    torch.manual_seed(-1)

    env = NormalizedEnv(env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    rospy.loginfo("State dim: {}, Action dim: {}".format(
        state_dim, action_dim))

    noise = GaussianActionNoise(mu=numpy.zeros(action_dim),
                                sigma=numpy.array([0.2, 0.2]))

    critic = Critic(state_dim, action_dim).to(device)
    actor = Actor(state_dim, action_dim).to(device)

    if load_actor != "":
        actor.load_state_dict(torch.load(os.path.join(result_dir, load_actor)))

    if load_critic != "":
        critic.load_state_dict(
            torch.load(os.path.join(result_dir, load_critic)))

    target_critic = Critic(state_dim, action_dim).to(device)
    target_actor = Actor(state_dim, action_dim).to(device)

    for target_param, param in zip(target_critic.parameters(),
                                   critic.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_actor.parameters(),
                                   actor.parameters()):
        target_param.data.copy_(param.data)

    q_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)
    policy_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)

    MSE = nn.MSELoss()

    memory = ReplayBuffer(buffer_size)

    plot_reward = []
    plot_policy = []
    plot_q = []
    plot_steps = []

    best_reward = -numpy.inf
    saved_reward = -numpy.inf
    saved_ep = 0
    average_reward = 0
    global_step = 0
    for episode in range(nepisodes):
        s = deepcopy(env.reset())

        ep_reward = 0.
        ep_q_value = 0.
        step = 0

        total_policy_loss = 0.0
        total_q_loss = 0.0

        eta *= eta_decay
        terminal = False
        while not terminal:
            global_step += 1
            step += 1
            epsilon -= epsilon_decay
            a = actor.get_action(s)

            if numpy.random.rand() < eta:
                a = numpy.random.uniform(-1, 1, action_dim)
                rospy.loginfo("CHOSE RANDOM ACTION {}".format(a))
            else:
                n = noise()
                rospy.loginfo("LEARNED ACTION {}, NOISE {}".format(a, n))
                a += n * max(0, epsilon)

            a = numpy.clip(a, -1, 1)  # normalization by normalized env wrapper
            s2, reward, terminal, info = env.step(a)

            memory.add(s, a, reward, terminal, s2)

            # keep adding experiences until batch size is reached
            if memory.count() > buffer_start:

                s_batch, a_batch, r_batch, t_batch, s2_batch = memory.sample(
                    batch_size)

                s_batch = torch.FloatTensor(s_batch).to(device)
                a_batch = torch.FloatTensor(a_batch).to(device)
                r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
                t_batch = torch.FloatTensor(
                    numpy.float32(t_batch)).unsqueeze(1).to(device)
                s2_batch = torch.FloatTensor(s2_batch).to(device)

                # critic loss
                a2_batch = target_actor(s2_batch)
                target_q = target_critic(s2_batch, a2_batch)
                y = r_batch + (1.0 - t_batch) * gamma * target_q.detach()
                q = critic(s_batch, a_batch)

                q_optimizer.zero_grad()
                q_loss = MSE(q, y)
                total_q_loss += q_loss.item()
                q_loss.backward()
                q_optimizer.step()

                # actor loss
                policy_optimizer.zero_grad()
                policy_loss = -critic(s_batch, actor(s_batch))
                policy_loss = policy_loss.mean()
                total_policy_loss += policy_loss.item()
                policy_loss.backward()
                policy_optimizer.step()

                # soft update frozen target networks
                for target_param, param in zip(target_critic.parameters(),
                                               critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - tau) +
                                            param.data * tau)

                for target_param, param in zip(target_actor.parameters(),
                                               actor.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - tau) +
                                            param.data * tau)

            s = deepcopy(s2)
            ep_reward += reward

        plot_reward.append(ep_reward)
        plot_policy.append(total_policy_loss)
        plot_q.append(total_q_loss)
        plot_steps.append(step + 1)

        average_reward += ep_reward

        if (episode % print_every) == (print_every - 1):
            # make sure path to results exists
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            torch.save(actor.state_dict(),
                       os.path.join(result_dir, 'best_actor_sailboat.pickle'))
            torch.save(critic.state_dict(),
                       os.path.join(result_dir, 'best_critic_sailboat.pickle'))
            best_reward = ep_reward
            saved_reward = ep_reward
            saved_ep = episode + 1

            fig, axs = plt.subplots(4, 1, sharex=True)
            axs[0].set_title("Episode Rewards")
            axs[0].plot(plot_reward, 'g-')
            axs[1].set_title('Policy Loss')
            axs[1].plot(plot_policy, 'r-')
            axs[2].set_title('Q Loss')
            axs[2].plot(plot_q, 'r-')
            axs[3].set_title('Steps per Episode')
            axs[3].plot(plot_steps, 'b-')
            plt.tight_layout()

            # make sure path to results exists
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            plt.savefig(
                os.path.join(result_dir, 'episode-{}.pdf'.format(episode)))

            rospy.loginfo(
                '[%6d episode, %8d total steps] average reward for past {} iterations: %.3f'
                .format(print_every) %
                (episode + 1, global_step, average_reward / print_every))

            rospy.loginfo(
                'Last model saved with reward: {:.2f}, at episode {}'.format(
                    saved_reward, saved_ep))

            average_reward = 0

    env.close()