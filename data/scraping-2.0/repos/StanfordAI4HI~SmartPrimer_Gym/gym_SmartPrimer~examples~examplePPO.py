import gym
from gym import wrappers
import numpy as np
from rlgraph.agents import Agent
from rlgraph.environments import OpenAIGymEnv
from rlgraph.execution import SingleThreadedWorker
import json
import os

### NOTE ###
#if you want to use PPO that never selects the same hint twice sequentially, use the ppo_agentSmartPrimer.py
#file in gym_SmartPrimer/agents/. Copy that file and place it into the rlraph/agents directory, under
# the name 'ppo_agent' (replace the old one)

np.random.seed(2)

#configure the agent settings in this file
agent_config_path = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/gym_SmartPrimer/agents/ppoSmartPrimer_config.json'
#agent_config_path = 'gym_SmartPrimer/agents/ppoSmartPrimer_config.json'

with open(agent_config_path, 'rt') as fp:
	agent_config = json.load(fp)

#create the environment
env = OpenAIGymEnv.from_spec({
        "type": "openai",
        "gym_env": 'gym_SmartPrimer:SmartPrimer-realistic-v2'
    })

#retreive the agent from RLgraph
agent = Agent.from_spec(
        agent_config,
        state_space=env.state_space,
        action_space=env.action_space
    )

#define number of children to simulate
episode_count = 3000

episode_returns = []
def episode_finished_callback(episode_return, duration, timesteps, *args, **kwargs):
	episode_returns.append(episode_return)
	if len(episode_returns) % 100 == 0:
		print("Episode {} finished: reward={:.2f}, average reward={:.2f}.".format(
			len(episode_returns), episode_return, np.mean(episode_returns[-100:])
		))

# create the worker
worker = SingleThreadedWorker(env_spec=lambda: env, agent=agent, render=False, worker_executes_preprocessing=False,
                                  episode_finish_callback=episode_finished_callback)

# Use exploration is true for training, false for evaluation.
worker.execute_episodes(episode_count, use_exploration=True)

#make the plots
env.gym_env.render()
