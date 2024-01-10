import os
import argparse

import numpy as np

from pommerman.agents import BaseAgent, SimpleAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

from tensorflow.keras.utils import to_categorical


class StoppingAgent(BaseAgent):
    def act(self, obs, action_space):
        # 0 stop, 1 up, 2 down, 3 left, 4 right, 5 bom
        return 0


def make_np_float(feature):
    return np.array(feature).astype(np.float32)


def featurize(obs):
    # 自分は10、敵は11に変更 (11, 11, 12)
    board = obs['board']
    board = np.where(13 == board, 10, board)
    board = np.where(10 < board, 11, board)
    nb_classes = 12
    board = to_categorical(board, nb_classes).astype(np.float32)
    return board


class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass


class WrappedEnv(OpenAIGym):
    def __init__(self, gym, agent, visualize=False):
        self.gym = gym
        self.agent = agent
        self.visualize = visualize

        self.episode_num = 1

        self.episode_reward = 0
        self.ammo = 1
        self.blast_strength = 2
        self.can_kick = False
        self.breakable_wall_count = 36
        self.bom_count = 0
        self.set_bom_count = 0

    def execute(self, action):
        if self.visualize:
            self.gym.render()

        actions = self.unflatten_action(action=action)

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]

        target_obs = obs[self.gym.training_agent]

        # obsの詳細
        # https://github.com/MultiAgentLearning/playground/tree/master/pommerman

        if not terminal:
            # 爆弾獲得
            if self.ammo < target_obs['ammo']:
                agent_reward += 0.2
            self.ammo = target_obs['ammo']

            # 火力獲得
            if self.blast_strength < target_obs['blast_strength']:
                agent_reward += 0.2
            self.blast_strength = target_obs['blast_strength']

            # キック獲得
            if self.can_kick is False and target_obs['can_kick'] is True:
                agent_reward += 0.2
            self.can_kick = target_obs['can_kick']

            # 爆弾数リワード
            # bom_count = np.sum(target_obs['board'] == 6)
            # agent_reward += bom_count * 0.1

            # # 爆弾設置回数
            # bom_count = np.sum(target_obs['board'] == 6)
            # if self.bom_count < bom_count:
            #     self.set_bom_count += 1
            #     agent_reward += 0.05
            # self.bom_count = bom_count

            # # 爆弾置けるのに置いていない
            # if not target_obs['ammo'] == self.bom_count:
            #     agent_reward -= 0.2

            # # 壁破壊
            # breakable_wall_count = np.sum(target_obs['board'] == 2)
            # if breakable_wall_count < self.breakable_wall_count:
            #     # max_reward = 最大報酬
            #     max_reward = 1
            #     break_wall_count = self.breakable_wall_count - breakable_wall_count
            #     break_wall_reward = 1 / 4
            #     agent_reward += max_reward * break_wall_count * break_wall_reward
            #     # agent_reward += max_reward
            # self.breakable_wall_count = breakable_wall_count

            # # 生存ペナルティ
            # agent_reward += -0.001

            # 生存step数ペナルティ
            agent_reward += target_obs['step_count'] * -0.0001

            # # 生存step数
            # agent_reward += target_obs['step_count'] * 0.0001

            # # 死んだ敵の数
            # dead_players_count = 4 - len(target_obs['alive'])
            # agent_reward += dead_players_count * 0.5

        self.episode_reward += agent_reward

        if terminal:
            print(f'episode {self.episode_num}, episode reward = {self.episode_reward}')
            self.episode_num += 1

            self.episode_reward = 0
            self.ammo = 1
            self.blast_strength = 2
            self.can_kick = False
            self.breakable_wall_count = 36
            self.bom_count = 0
            self.set_bom_count = 0

        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = featurize(obs[3])
        return agent_obs


def main(args):
    version = 'v1'
    episodes = args.episodes
    visualize = args.visualize

    config = ffa_v0_fast_env()
    env = Pomme(**config["env_kwargs"])
    env.seed(0)

    agent = PPOAgent(
        states=dict(type='float', shape=(11, 11, 12)),
        actions=dict(type='int', num_actions=env.action_space.n),
        network=[
            # (9, 9, 12)
            dict(type='conv2d', size=12, window=3, stride=1),
            # (7, 7, 8)
            dict(type='conv2d', size=8, window=3, stride=1),
            # (5, 5, 4)
            dict(type='conv2d', size=4, window=3, stride=1),
            # (100)
            dict(type='flatten'),
            dict(type='dense', size=64, activation='relu'),
            dict(type='dense', size=16, activation='relu'),
        ],
        batching_capacity=1000,
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-4
        )
    )

    if os.path.exists(os.path.join('models', version, 'checkpoint')):
        agent.restore_model(directory=os.path.join('models', version))

    agents = []
    for agent_id in range(3):
        # agents.append(RandomAgent(config["agent"](agent_id, config["game_type"])))
        # agents.append(StoppingAgent(config["agent"](agent_id, config["game_type"])))
        agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

    agent_id += 1
    agents.append(TensorforceAgent(config["agent"](agent_id, config["game_type"])))
    env.set_agents(agents)
    env.set_training_agent(agents[-1].agent_id)
    env.set_init_game_state(None)

    wrapped_env = WrappedEnv(env, agent, visualize)
    runner = Runner(agent=agent, environment=wrapped_env)

    try:
        runner.run(episodes=episodes, max_episode_timesteps=100)
    except Exception as e:
        raise e
    finally:
        agent.save_model(directory=os.path.join('models', version, 'agent'))

    win_count = len(list(filter(lambda reward: reward == 1, runner.episode_rewards)))
    print('Stats: ')
    print(f'  runner.episode_rewards = {runner.episode_rewards}')
    print(f'  win count = {win_count}')

    try:
        runner.close()
    except AttributeError as e:
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--episodes', type=int, default=100)

    args, _ = parser.parse_known_args()
    main(args)
