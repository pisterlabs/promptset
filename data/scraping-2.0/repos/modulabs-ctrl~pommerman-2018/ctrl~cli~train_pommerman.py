# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")

"""Train an agent with TensorForce.

Call this with a config, a game, and a list of agents, one of which should be a
tensorforce agent. The script will start separate threads to operate the agents
and then report back the result.

An example with all three simple agents running ffa:
python train_with_tensorforce.py \
 --agents=tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent \
 --config=PommeFFACompetition-v0
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

CLIENT = docker.from_env()

DEFAULT_REWARDS = [ 
    1.01, # 0. 승리
   -1.01, # 1. 패배
    0.01, # 2. 무승부
   -0.01, # 3. 아무 행동도 하지 않으면 타임 스텝마다
    0.01, # 4. 폭탄을 설치하면 해당 타임스텝
    0.01, # 5. 계속 움직이면 해당 타임스텝
    0.01, # 6. 아이템 (폭탄범위) 최초로 먹으면
    0.01, # 7. 아이템 (킥) 최초로 먹으면
    0.01  # 8. 아이템 (탄창) 최초로 먹으면
]
RES_WIN = 0
RES_LOSE = 1
RES_DRAW = 2
ACT_SLEEP = 3
ACT_BOMB = 4
ACT_OTHER = 5
ITEM_BLAST = 6
ITEM_KICK = 7
ITEM_AMMO = 8

STR_WINNER='Winner' # :thumbs_up_light_skin_tone:'
STR_LOSER='Loser' # :thumbs_down_light_skin_tone:'
STR_SLEEP='Sleep'
STR_STAY='Stay'
STR_UP='Up'
STR_LEFT='Left'
STR_DOWN='Down'
STR_RIGHT='Right'
STR_BOMBSET='BombSet' # :bomb:'
STR_BLAST='ItemBlast' # :cookie:'
STR_KICK='ItemKick' # :egg:'
STR_AMMO='ItemAmmo' # :rice:'

def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]

class WrappedEnv(OpenAIGym):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''
    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize
        self.old_position = None
        self.prev_position = None
        self.curr_position = None
        self.timestep = 0
        self.episode = 0
        self.has_blast_strength = False
        self.has_can_kick = False
        self.has_ammo = False
        self.tmp_reward = 0.0
        self.res_reward = 0.0
        self.accu_bombset = 1.0
        self.act_history = []
        self.render = False
        self.rewards = DEFAULT_REWARDS
        print(f'Episode [{self.episode:03}], Timestep [{self.timestep:03}] initialized.')

    def set_render(self, render):
        self.render = render

    def set_rewards(self, custom_rewards):
        self.rewards = [ float(reward.strip()) for reward in custom_rewards.split(',') ]
        print(self.rewards)
    
    def shaping_reward(self, agent_id, agent_obs, agent_reward, agent_action):
        import emoji
        import numpy as np
        self.timestep += 1
        self.agent_board = agent_obs['board']
        self.curr_position = np.where(self.agent_board == agent_id)
        self.tmp_reward = 0.0
        actions = []

        if agent_reward == 1:
            actions.append(emoji.emojize(STR_WINNER))
            self.tmp_reward += self.rewards[RES_WIN]
        if agent_reward == -1:
            actions.append(emoji.emojize(STR_LOSER))
            self.tmp_reward += self.rewards[RES_LOSE]
        if agent_reward == 0:
            # actions.append("Draw")
            self.tmp_reward += self.rewards[RES_DRAW]

        if self.prev_position != None and self.prev_position == self.curr_position and self.old_position == self.prev_position:
            actions.append(emoji.emojize(STR_SLEEP))
            self.tmp_reward += self.rewards[ACT_SLEEP]
        elif agent_action == 0:
            actions.append(emoji.emojize(STR_STAY))
            self.tmp_reward += self.rewards[ACT_OTHER]
        elif agent_action == 1:
            actions.append(emoji.emojize(STR_UP))
            self.tmp_reward += self.rewards[ACT_OTHER]
        elif agent_action == 2:
            actions.append(emoji.emojize(STR_LEFT))
            self.tmp_reward += self.rewards[ACT_OTHER]
        elif agent_action == 3:
            actions.append(emoji.emojize(STR_DOWN))
            self.tmp_reward += self.rewards[ACT_OTHER]
        elif agent_action == 4:
            actions.append(emoji.emojize(STR_RIGHT))
            self.tmp_reward += self.rewards[ACT_OTHER]
        elif agent_action == 5:
            actions.append(emoji.emojize(STR_BOMBSET))
            self.tmp_reward += self.rewards[ACT_BOMB] * self.accu_bombset
            self.accu_bombset += 0.2

        if not self.has_blast_strength and int(agent_obs['blast_strength']) > 2:
            actions.append(emoji.emojize(STR_BLAST))
            self.tmp_reward += self.rewards[ITEM_BLAST]
            self.has_blast_strength = True
        if not self.has_can_kick and agent_obs['can_kick'] == True:
            actions.append(emoji.emojize(STR_KICK))
            self.tmp_reward += self.rewards[ITEM_KICK]
            self.has_can_kick = True
        if not self.has_ammo and int(agent_obs['ammo']) > 1:
            actions.append(emoji.emojize(STR_AMMO))
            self.tmp_reward += self.rewards[ITEM_AMMO]
            self.has_ammo = True

        self.res_reward += self.tmp_reward
        self.act_history += actions
        # 렌더링 하는 경우에만 자세한 리워드를 출력한다.
        if self.render:
            print(f'Episode [{self.episode:03}], Timestep [{self.timestep:03}] got reward {round(self.res_reward, 2)} [{actions}]')

        self.old_position = self.prev_position
        self.prev_position = self.curr_position
        
        # feature 네트워크

        return self.tmp_reward

    def execute(self, action):
        if self.visualize:
            self.gym.render()

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, action)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent])

        agent_id = self.gym.training_agent + 10
        agent_reward = reward[self.gym.training_agent]
        agent_action = all_actions[self.gym.training_agent]
        agent_obs = obs[self.gym.training_agent]
        modified_reward = self.shaping_reward(agent_id, agent_obs, agent_reward, agent_action)

        return agent_state, terminal, modified_reward

    '''Reset method is called when every episode starts'''
    def reset(self):
        hist = self.act_history
        item_count = hist.count(STR_AMMO) + hist.count(STR_BLAST) + hist.count(STR_KICK)
        bomb_count = hist.count(STR_BOMBSET)
        move_count = hist.count(STR_UP) + hist.count(STR_DOWN) + hist.count(STR_LEFT) + hist.count(STR_RIGHT)
        stop_count = hist.count(STR_SLEEP) + hist.count(STR_STAY)
        history = "BombSet({}), ItemGot({}), Move({}), Stay({})".format(bomb_count, item_count, move_count, stop_count)

        print(f'Episode [{self.episode:03}], Timestep [{self.timestep:03}] reward {round(self.res_reward,2)} history {history}.')
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[3])
        self.timestep = 0
        self.episode += 1
        self.tmp_reward = 0.0
        self.res_reward = 0.0
        self.accu_bombset = 1.0
        self.act_history = []
        self.has_blast_strength = False
        self.has_can_kick = False
        self.has_ammo = False
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
        "--rewards",
        default=DEFAULT_REWARDS,
        help="Shaping of rewards"
    )
    args = parser.parse_args()

    config = args.config
    # record_pngs_dir = args.record_pngs_dir
    # record_json_dir = args.record_json_dir
    # agent_env_vars = args.agent_env_vars
    game_state_file = args.game_state_file
    checkpoint = args.checkpoint
    num_of_episodes = int(args.num_of_episodes)
    max_timesteps = int(args.max_timesteps)
    custom_rewards = args.rewards

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
    wrapped_env.set_rewards(custom_rewards)

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