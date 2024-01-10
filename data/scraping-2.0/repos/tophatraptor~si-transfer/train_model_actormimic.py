#!/usr/bin/env python3
import gym
import ptan
import argparse

import torch
from torch import nn
import torch.optim as optim
import torch.multiprocessing as mp

import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

from lib import dqn_model, common

from collections import namedtuple, deque
import csv
import numpy as np
import os

PLAY_STEPS = 4

# one single experience step
Experience_AM = namedtuple('Experience_AM', ['state', 'action', 'reward', 'done', 'env'])

class ExperienceSource_AM:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """
    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        :param vectorized: support of vectorized envs from OpenAI universe
        """
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, ptan.agent.BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized

    def __iter__(self):
        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        env_lens = []
        for env in self.pool:
            obs = env.reset()
            # if the environment is vectorized, all it's output is lists of results.
            # Details are here: https://github.com/openai/universe/blob/master/doc/env_semantics.rst
            if self.vectorized:
                obs_len = len(obs)
                states.extend(obs)
            else:
                obs_len = 1
                states.append(obs)
            env_lens.append(obs_len)

            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count))
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(self.agent.initial_state())

        iter_idx = 0
        while True:
            actions = [None] * len(states)
            states_input = []
            states_indices = []
            for idx, state in enumerate(states):
                if state is None:
                    actions[idx] = self.pool[0].action_space.sample()  # assume that all envs are from the same family
                else:
                    states_input.append(state)
                    states_indices.append(idx)
            if states_input:
                states_actions, new_agent_states = self.agent(states_input, agent_states)
                for idx, action in enumerate(states_actions):
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]
            grouped_actions = ptan.experience._group_list(actions, env_lens)

            global_ofs = 0
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    next_state_n, r_n, is_done_n, _ = env.step(action_n)
                else:
                    next_state, r, is_done, _ = env.step(action_n[0])
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]

                #### This is the addition
                env_name = env.unwrapped.spec.id

                for ofs, (action, next_state, r, is_done) in enumerate(zip(action_n, next_state_n, r_n, is_done_n)):
                    idx = global_ofs + ofs
                    state = states[idx]
                    history = histories[idx]

                    cur_rewards[idx] += r
                    cur_steps[idx] += 1
                    if state is not None:
                        history.append(Experience_AM(state=state, action=action, reward=r, done=is_done, env=env_name))
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    states[idx] = next_state
                    if is_done:
                        # generate tail of history
                        while len(history) >= 1:
                            yield tuple(history)
                            history.popleft()
                        self.total_rewards.append(cur_rewards[idx])
                        self.total_steps.append(cur_steps[idx])
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        # vectorized envs are reset automatically
                        states[idx] = env.reset() if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_state()
                        history.clear()
                global_ofs += len(action_n)
            iter_idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


# those entries are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast_AM = namedtuple('ExperienceFirstLast_AM', ('state', 'action', 'reward', 'last_state', 'env'))


class ExperienceSourceFirstLast_AM(ExperienceSource_AM):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1, vectorized=False):
        assert isinstance(gamma, float)
        super(ExperienceSourceFirstLast_AM, self).__init__(env, agent, steps_count+1, steps_delta, vectorized=vectorized)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast_AM, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast_AM(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state, env=exp[0].env)


def play_func(params, net, cuda, exp_queue, device_id):
    """
    The paper suggests sampling the actions from the learner net, so that requires little change from the multienv implementation.

    *** There is a reason that it reinitializes the envs in this function that has to do with parallelization ***
    """
    run_name = params['run_name']
    if 'max_games' not in params:
        max_games = 16000
    else:
        max_games = params['max_games']

    envSI = gym.make('SpaceInvadersNoFrameskip-v4')
    envSI = ptan.common.wrappers.wrap_dqn(envSI)

    envDA = gym.make('DemonAttackNoFrameskip-v4')
    envDA = ptan.common.wrappers.wrap_dqn(envDA)

    device = torch.device("cuda:{}".format(device_id) if cuda else "cpu")

    writer = SummaryWriter(comment="-" + run_name + "-03_parallel")

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ExperienceSourceFirstLast_AM([envSI, envDA], agent, gamma=params['gamma'], steps_count=1)
    exp_source_iter = iter(exp_source)

    fh = open('mimic_models/{}_metadata.csv'.format(run_name), 'w')
    out_csv = csv.writer(fh)

    frame_idx = 0
    game_idx = 1
    model_count = 0
    model_stats = []
    mean_rewards = []
    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            epsilon_tracker.frame(frame_idx)
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                status, num_games, mean_reward, epsilon_str = reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon)
                mean_rewards.append(mean_reward)
                if status:
                    break
                if game_idx and (game_idx % 500 == 0):
                    # write to disk
                    print("Saving model...")
                    model_name = 'mimic_models/{}_{}.pth'.format(run_name, game_idx)
                    net.to(torch.device('cpu'))
                    torch.save(net, model_name)
                    net.to(device)
                    new_row = [model_name, num_games, mean_reward, epsilon_str]
                    out_csv.writerow(new_row)
                    np.savetxt('mimic_models/{}_reward.txt'.format(run_name), np.array(mean_rewards))
                if game_idx == max_games:
                    break
                game_idx += 1

    print("Saving final model...")
    model_name = 'mimic_models/{}_{}.pth'.format(run_name, game_idx)
    net.to(torch.device('cpu'))
    torch.save(net, model_name)
    net.to(device)
    new_row = [model_name, num_games, mean_reward, epsilon_str]
    out_csv.writerow(new_row)
    np.savetxt('mimic_models/{}_reward.txt'.format(run_name), np.array(mean_rewards))
    # plt.figure(figsize=(16, 9))
    # plt.tight_layout()
    # plt.title('Reward vs time, {}'.format(run_name))
    # plt.xlabel('Iteration')
    # plt.ylabel('Reward')
    # ys = np.array(mean_rewards)
    # plt.plot(ys, c='r')
    # plt.savefig('mimic_models/{}_reward.png'.format(run_name))
    # plt.close()
    fh.close()

    exp_queue.put(None)


if __name__ == "__main__":
    """
    This method attempts to build a generalized model not from the raw games but from two expert models trained on individual games.

    It does so by training a new network to replicate the output Q values of the multiple expert models given the same input.
    Therefore, the loss is given by the MSE of the softmax of the final activations + MSE of the final hidden layer.

    See https://arxiv.org/pdf/1511.06342.pdf

    Adds a slight edit to the Experience objects noting which game they are from so as to use the correct expert
    """
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--cuda_id", default=0, help="CUDA ID of device")
    parser.add_argument("--si", help="Path to space invaders master model", required=True)
    parser.add_argument('--da', help='Paths to demon attack master model', required=True)
    parser.add_argument('--env', help='Environment to load', required=True)
    parser.add_argument('--beta', help='Balance of Policy vs Hidden Loss', default=1)
    args = parser.parse_args()
    cuda_id = args.cuda_id
    params = common.HYPERPARAMS[args.env]
    params['batch_size'] *= PLAY_STEPS
    device_str = "cuda:{}".format(cuda_id) if args.cuda else "cpu"
    print("Using device: {}".format(device_str))
    device = torch.device(device_str)

    if not os.path.exists('mimic_models'):
        os.makedirs('mimic_models')

    envSI = gym.make('SpaceInvadersNoFrameskip-v4')
    envSI = ptan.common.wrappers.wrap_dqn(envSI)

    envDA = gym.make('DemonAttackNoFrameskip-v4')
    envDA = ptan.common.wrappers.wrap_dqn(envDA)

    assert envSI.action_space.n == envDA.action_space.n, "Different Action Space Lengths"
    assert envSI.observation_space.shape == envDA.observation_space.shape, "Different Obs. Space Shapes"

    print("Loaded Environments: {}l {}".format(envSI.unwrapped.spec.id, envDA.unwrapped.spec.id))

    expertSI = dqn_model.DQN(envSI.observation_space.shape, envSI.action_space.n)
    expertSI.load_state_dict(torch.load(args.si, map_location=device).state_dict())
    expertSI_hidden = dqn_model.DQN_Hidden(envSI.observation_space.shape, envSI.action_space.n, expertSI).to(device)
    expertSI = expertSI.to(device)
    expertSI.eval()
    expertSI_hidden.eval()

    expertDA = dqn_model.DQN(envSI.observation_space.shape, envSI.action_space.n)
    expertDA.load_state_dict(torch.load(args.da, map_location=device).state_dict())
    expertDA_hidden = dqn_model.DQN_Hidden(envSI.observation_space.shape, envSI.action_space.n, expertDA).to(device)
    expertDA = expertDA.to(device)
    expertDA.eval()
    expertDA_hidden.eval()

    name_to_expert = {envSI.unwrapped.spec.id : [expertSI, expertSI_hidden] , envDA.unwrapped.spec.id : [expertDA, expertDA_hidden]}

    # This net will attempt to learn the directly from the expert models, not the games. No target net needed
    net = dqn_model.DQN_AM(envSI.observation_space.shape, envSI.action_space.n).to(device)

    # After instantiating the model shape, we actually don't need these envs (will be recreated in the parallel function)
    del envSI
    del envDA

    # Now we want two buffers, one for each game, to keep the frames separate so we can use the correct expert model
    buffer = ptan.experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    exp_queue = mp.Queue(maxsize=PLAY_STEPS * 2)
    play_proc = mp.Process(target=play_func, args=(params, net, args.cuda, exp_queue, cuda_id))
    play_proc.start()

    frame_idx = 0

    while play_proc.is_alive():
        frame_idx += PLAY_STEPS
        for _ in range(PLAY_STEPS):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)

        if len(buffer) < 1000: #params['replay_initial']:
            continue
        # print("UPDATING GRAD")
        optimizer.zero_grad()
        batch = buffer.sample(params['batch_size'])
        loss_v = common.calc_loss_actormimic(batch, net, name_to_expert, beta=args.beta, cuda=args.cuda, cuda_async=True, cuda_id=cuda_id)
        loss_v.backward()
        optimizer.step()
