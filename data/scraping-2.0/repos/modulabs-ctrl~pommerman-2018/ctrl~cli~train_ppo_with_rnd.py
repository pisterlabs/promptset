# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")

"""Train an agent with TensorForce.
python ctrl/cli/train_ppo_with_rnd.py \
--agents=tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent \
--num_of_episodes=1000 --max_timesteps=1000 --config=PommeFFAFast-v0  --render --simulation 2000
"""

import atexit
import functools
import os, sys

import argparse
import docker
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import gym

sys.path.append('.')

from pommerman import helpers, make
from pommerman.agents import TensorForceAgent
from ctrl.agents import TensorForcePpoAgent

from collections import deque
from copy import deepcopy

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

SEED = 5
BATCH_SIZE = 255
LR = 0.0001
LE = 1e-5
SIZE = 372
EPS = np.finfo(np.float).eps
INTRINSIC_CLIP=0.0001

# set device
use_cuda = torch.cuda.is_available()
print('cuda:', use_cuda)
device = torch.device('cuda' if use_cuda else 'cpu')

# random seed
np.random.seed(SEED)
torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed_all(SEED)

# ppo settings
CLIENT = docker.from_env()

def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]

class RandomNet(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(obs_space, SIZE),
            nn.SELU()
        )

        self.fc = nn.Sequential(
            nn.Linear(SIZE, SIZE*2)
        )

    def forward(self, x):
        out = self.head(x)
        obs_feature = self.fc(out).reshape(out.shape[0], -1)

        return obs_feature


class PredictNet(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(obs_space, SIZE),
            nn.SELU()
        )

        self.fc = nn.Sequential(
            nn.Linear(SIZE, SIZE),
            nn.SELU(),
            nn.Linear(SIZE, SIZE),
            nn.SELU(),
            nn.Linear(SIZE, SIZE*2)
        )

    def forward(self, x):
        out = self.head(x)
        obs_feature = self.fc(out).reshape(out.shape[0], -1)

        return obs_feature

class WrappedEnv(OpenAIGym):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''
    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.action_space = gym.action_space.n
        self.visualize = visualize
        self.timestep = 0
        self.global_ts = 0
        self.episode = 0
        self.res_reward = 0.0
        self.render = False
        # curiosity properties
        self.obs_space = gym.observation_space.shape[0]
        self.threshold_of_simulation = -1
        self.simulated = 0
        self.obs_memory = []
        self.max_timesteps = 0
        self.rep_memory = None
        self.mean = 0.0
        self.std = 0.0
        self.pred_net = PredictNet(self.obs_space).to(device)
        self.rand_net = RandomNet(self.obs_space).to(device)
        self.pred_optim = torch.optim.Adam(self.pred_net.parameters(), lr=LR, eps=LE)

    def set_render(self, render):
        self.render = render

    def set_simulation(self, simulation):
        self.threshold_of_simulation = int(simulation)

    def set_max_timesteps(self, max_timesteps):
        self.max_timesteps = max_timesteps
        self.reset_replay_memory()

    def reset_replay_memory(self):
        self.rep_memory = deque(maxlen=self.max_timesteps)
        
    def get_norm_params_old(self, obs_memory):
        obses = [[] for _ in range(self.obs_space)]
        for obs in obs_memory:
            for j in range(self.obs_space):
                obses[j].append(obs[j])

        mean = np.zeros(self.obs_space, np.float32)
        std = np.zeros(self.obs_space, np.float32)
        for i, obs_ in enumerate(obses):
            mean[i] = np.mean(obs_)
            std[i] = np.std(obs_)
        print("get_norm_params : {}, {}".format(mean, std))
        return mean, std

    def get_norm_params(self, obs_memory):
        global obs_apace

        obses = [[] for _ in range(self.obs_space)]
        for obs in obs_memory:
            for j in range(self.obs_space):
                obses[j].append(obs[j])

        mean = np.zeros(self.obs_space, 'float')
        std = np.zeros(self.obs_space, 'float')
        for i, obs_ in enumerate(obses):
            mean[i] = np.mean(obs_)
            std[i] = np.std(obs_)
        obs_memory.clear()
        std = np.clip(std, a_min=EPS, a_max=None)
        print("mean:'{}', std:'{}'".format(mean, std))
        return mean, std

    def normalize_obs(self, label, obs, mean, std):
        means = [mean for _ in range(BATCH_SIZE)]
        stds = [std for _ in range(BATCH_SIZE)]
        mean = np.stack(means)
        std = np.stack(stds)
        norm_obs = (obs - mean) / std
        return np.clip(norm_obs, -1, 1)

    def calculate_reward_in(self, pred_net, rand_net, obs):
        norm_obs = self.normalize_obs("calculate", obs, self.mean, self.std)
        state = torch.tensor([norm_obs]).to(device).float()
        with torch.no_grad():
            pred_obs = pred_net(state)
            rand_obs = rand_net(state)
            reward = (pred_obs - rand_obs).pow(2).sum()
            clipped_reward = torch.clamp(reward, -1.0 * INTRINSIC_CLIP, 1.0 * INTRINSIC_CLIP)
        return clipped_reward.item()

    def execute(self, action):
        if self.visualize:
            self.gym.render()
        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, action)
        state, reward, terminal, _ = self.gym.step(all_actions)
        # state of t+1
        agent_state = self.gym.featurize(state[self.gym.training_agent])
        extrinsic_reward = reward[self.gym.training_agent]
        intrinsic_reward = 0.0
        # simulation 
        if self.simulated < self.threshold_of_simulation:
            self.obs_memory.append(agent_state)
            self.simulated += 1
        elif self.simulated == self.threshold_of_simulation:
            print("##### simulation has started #####")
            self.mean, self.std = self.get_norm_params(self.obs_memory)
            print("mean:{}, \n std:{} \n".format(self.mean, self.std))
            self.simulated += 1
            self.obs_memory = None
        else:
            intrinsic_reward = self.calculate_reward_in(self.pred_net, self.rand_net, agent_state)
            if np.isnan(intrinsic_reward):
                intrinsic_reward = 0.0
            # else:
            #     print("##### intrinsic_reward {} #####".format(intrinsic_reward))
        self.timestep += 1
        self.rep_memory.append(agent_state)
        agent_reward = extrinsic_reward + intrinsic_reward
        if self.timestep % 100 == 0:
            print("ts[{}/{}]: ext:'{}' + int:'{}' = rew:'{}'".format(self.timestep, self.global_ts, extrinsic_reward, intrinsic_reward, agent_reward))

        self.res_reward += agent_reward
        return agent_state, terminal, agent_reward

    def learn(self):
        self.pred_net.train()
        self.rand_net.train()

        dataloader = DataLoader(
            self.rep_memory,
            shuffle=True,
            batch_size=BATCH_SIZE,
            pin_memory=use_cuda
        )

        # training pred_net -- batch size 가 4라서 agent_state 는 4개의 튜플이다.
        for i, agent_state in enumerate(dataloader):
            obs = agent_state.detach().cpu().numpy()
            if obs.shape[0] != BATCH_SIZE: # shape 가 안 맞는 경우는 skip
                continue
            norm_state = self.normalize_obs("training", obs, self.mean, self.std)
            norm_batch = torch.tensor(norm_state).to(device).float()
            if i == 0:
                print("origi_norm: {}".format(obs[0]))
                print("norm_state: {}".format(norm_state))
                print("norm_batch: {}".format(norm_batch))

            pred_features = self.pred_net(norm_batch)
            rand_features = self.rand_net(norm_batch)

            f_loss = (pred_features - rand_features).pow(2).sum(dim=1).mean()

            if i == 0:
                print("pred_features: {}".format(pred_features))
                print("rand_features: {}".format(rand_features))
                print("loss_features: {}".format(pred_features - rand_features))

            self.pred_optim.zero_grad()
            f_loss.backward()
            self.pred_optim.step()

    '''Reset method is called when every episode starts'''
    def reset(self):
        if self.simulated > self.threshold_of_simulation:
            self.learn()
        self.global_ts += self.timestep
        self.reset_replay_memory()
        print(f'Episode [{self.episode:03}], Timestep [{self.timestep:03}/{self.global_ts}] reward {round(self.res_reward,2)}')
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[3])
        self.timestep = 0
        self.episode += 1
        return agent_obs

def create_ppo_agent(agent):
    if type(agent) == TensorForceAgent:
        print("create_ppo_agent({})".format(agent))
        return TensorForcePpoAgent()
    return agent

def main():
    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument("--game", default="pommerman", help="Game to choose.")
    parser.add_argument(
        "--config",
        default="PommeFFACompetition-v0",
        help="Configuration to execute. See env_ids in "
        "configs.py for options.")
    parser.add_argument(
        "--agents",
        default="tensorforce::ppo,test::agents.SimpleAgent,"
        "test::agents.SimpleAgent,test::agents.SimpleAgent",
        help="Comma delineated list of agent types and docker "
        "locations to run the agents.")
    parser.add_argument(
        "--agent_env_vars",
        help="Comma delineated list of agent environment vars "
        "to pass to Docker. This is only for the Docker Agent."
        " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
        "would send two arguments to Docker Agent 0 and one to"
        " Docker Agent 3.",
        default="")
    parser.add_argument(
        "--record_pngs_dir",
        default=None,
        help="Directory to record the PNGs of the game. "
        "Doesn't record if None.")
    parser.add_argument(
        "--record_json_dir",
        default=None,
        help="Directory to record the JSON representations of "
        "the game. Doesn't record if None.")
    parser.add_argument(
        "--render",
        default=False,
        action='store_true',
        help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--game_state_file",
        default=None,
        help="File from which to load game state. Defaults to "
        "None.")
    parser.add_argument(
        "--checkpoint",
        default="models/ppo",
        help="Directory where checkpoint file stored to."
    )
    parser.add_argument(
        "--num_of_episodes",
        default="10",
        help="Number of episodes"
    )
    parser.add_argument(
        "--max_timesteps",
        default="2000",
        help="Number of steps"
    )
    parser.add_argument(
        "--simulation",
        default=False,
        help="Number of simulations"
    )
    args = parser.parse_args()

    config = args.config
    game_state_file = args.game_state_file
    checkpoint = args.checkpoint
    num_of_episodes = int(args.num_of_episodes)
    max_timesteps = int(args.max_timesteps)
    simulation = args.simulation

    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.
    agents = [
        create_ppo_agent(helpers.make_agent_from_string(agent_string, agent_id + 1000))
        for agent_id, agent_string in enumerate(args.agents.split(","))
    ]

    env = make(config, agents, game_state_file)
    training_agent = None
    training_agent_id = None

    for agent in agents:
        if type(agent) == TensorForcePpoAgent:
            print("Ppo agent initiazlied : {}, {}".format(agent, type(agent)))
            training_agent = agent
            env.set_training_agent(agent.agent_id)
            training_agent_id = agent.agent_id
            break
        print("[{}] : id[{}]".format(agent, agent.agent_id))

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    learning_agent = training_agent.initialize(env)
    for agent in agents:
        if type(agent) == TensorForcePpoAgent:
            if agent.agent_id == training_agent_id:
                learning_agent = training_agent.initialize(env)
            else:
                agent.initialize(env)

    atexit.register(functools.partial(clean_up_agents, agents))
    wrapped_env = WrappedEnv(env, visualize=args.render)
    wrapped_env.set_render(args.render)
    wrapped_env.set_simulation(simulation)
    wrapped_env.set_max_timesteps(max_timesteps)

    runner = Runner(agent=learning_agent, environment=wrapped_env)
    runner.run(episodes=num_of_episodes, max_episode_timesteps=max_timesteps)
    print("Stats: ",
        runner.episode_rewards[-30:],
        runner.episode_timesteps,
        runner.episode_times)

    learning_agent.save_model(checkpoint)

    rewards = runner.episode_rewards
    import numpy as np
    mean = np.mean(rewards)
    print('last 30 rewards {}'.format(rewards[-30:]))
    print('mean of rewards {}'.format(mean))

    try:
        runner.close()
    except AttributeError as e:
        print(e)
        pass

if __name__ == "__main__":
    main()