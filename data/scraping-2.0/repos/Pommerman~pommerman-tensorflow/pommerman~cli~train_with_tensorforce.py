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
import os

import argparse
import docker
from tensorforce.execution import ParallelRunner
from tensorforce.execution import Runner
#from pommerman.runner import ExperimentRunner
from tensorforce.environments.openai_gym import OpenAIGym
import gym

from pommerman import helpers, make
from pommerman.agents import TensorForceAgent

import timeit
import pickle
import matplotlib.pyplot as plt

CLIENT = docker.from_env()
save_name = 'ppo'
def save_obj(obj):
    with open('./saved_models/'+ save_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj():
    try:
        with open('./saved_models/' + save_name + '.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return []

def load_model(agent, path):
    if(os.path.exists(path) == False):
        os.mkdir(path)
        return []
    
    # agent.restore_model(path)
    return load_obj()

def save_model(agent, path, hist, addTimestamp):
    # agent.save_model(path, addTimestamp)
    save_obj(hist)

def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]


class WrappedEnv(OpenAIGym):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''

    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize
        self.thread=None
    def execute(self, actions):
        if self.visualize:
            self.gym.render()

        actions = self.unflatten_action(action=actions)

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[3])
        return agent_obs

    @staticmethod
    def conv_action(action):
        if not isinstance(action, dict):
            return action
        elif all(name.startswith('gymmdc') for name in action) or \
                all(name.startswith('gymbox') for name in action) or \
                all(name.startswith('gymtpl') for name in action):
            space_type = next(iter(action))[:6]
            actions = list()
            n = 0
            while True:
                if any(name.startswith(space_type + str(n) + '-') for name in action):
                    inner_action = {
                        name[name.index('-') + 1:] for name, inner_action in action.items()
                        if name.startswith(space_type + str(n))
                    }
                    actions.append(OpenAIGym.unflatten_action(action=inner_action))
                elif any(name == space_type + str(n) for name in action):
                    actions.append(action[space_type + str(n)])
                else:
                    break
                n += 1
            return tuple(actions)
        else:
            actions = dict()
            for name, action in action.items():
                if '-' in name:
                    name, inner_name = name.split('-', 1)
                    if name not in actions:
                        actions[name] = dict()
                    actions[name][inner_name] = action
                else:
                    actions[name] = action
            for name, action in actions.items():
                if isinstance(action, dict):
                    actions[name] = OpenAIGym.unflatten_action(action=action)
            return actions



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
        "--num_procs",
        default=12,
        type=int,
        help="Number of parallel threads to run. Defaults to 12."
    )
    args = parser.parse_args()

    config = args.config
    record_pngs_dir = args.record_pngs_dir
    record_json_dir = args.record_json_dir
    agent_env_vars = args.agent_env_vars
    game_state_file = args.game_state_file
    num_procs = args.num_procs

    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.
    agents = [
        helpers.make_agent_from_string(agent_string, agent_id + 1000)
        for agent_id, agent_string in enumerate(args.agents.split(","))
    ]

    env = make(config, agents, game_state_file)
    training_agent = None

    for agent in agents:
        if type(agent) == TensorForceAgent:
            training_agent = agent
            env.set_training_agent(agent.agent_id)
            break

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    # Create a Proximal Policy Optimization agent
    agent = training_agent.initialize(env, num_procs,
                summarizer={'directory': 'tensorforce_agent', 'labels': 'all'},
                saver={'directory': './saved_models', 'filename': 'ppo'})

    hist = load_model(agent, './saved_models')


    atexit.register(functools.partial(clean_up_agents, agents))

    wrapped_envs=[]
    for i in range(num_procs):
        wrapped_envs.append(WrappedEnv(env, visualize=args.render))

    runner_time = timeit.default_timer()

    
    for i in range(1):
        runner = ParallelRunner(agent=agent, environments=wrapped_envs)
        runner.run(num_episodes=1000, max_episode_timesteps=2000)

        print("Stats: ", runner.episode_rewards, runner.episode_timesteps,
            runner.episode_seconds)
        hist = {
            "episode_rewards": hist.episode_rewards.extend(runner.episode_rewards),
            "episode_timesteps": hist.episode_timesteps.extend(runner.episode_timesteps),
            "episode_times": hist.episode_seconds.extend(runner.episode_seconds)
        }

    print('Runner time: ', timeit.default_timer() - runner_time)

    save_model(agent, 'saved_models\\ppo', hist, True)
    
    plt.plot(runner.episode_rewards)
    plt.show()
    try:
        runner.close()
    except AttributeError as e:
        pass


if __name__ == "__main__":
    main()
