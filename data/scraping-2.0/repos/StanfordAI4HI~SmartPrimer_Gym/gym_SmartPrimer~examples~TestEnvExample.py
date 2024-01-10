import gym
from gym import wrappers
import numpy as np
from rlgraph.agents import Agent
from rlgraph.environments import OpenAIGymEnv
import json
import os
import pickle
import argparse
import copy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='example')
parser.add_argument('--seed', type=int, default=3,
                    help='numpy seed ')
parser.add_argument('--time', type=int, default=5,
                    help='numpy seed ')
args = parser.parse_args()

np.random.seed(args.seed)


env = OpenAIGymEnv.from_spec({
	"type": "openai",
	"gym_env": 'gym_SmartPrimer:TestEnv-v0'
})

# configure the agent settings in this file
agent_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'agents/ppoSmartPrimer_config.json')

with open(agent_config_path, 'rt') as fp:
	agent_config = json.load(fp)

# retreive the agent from RLgraph
agent = Agent.from_spec(
	agent_config,
	state_space=env.state_space,
	action_space=env.action_space
)

# define number of children to simulate
episode_count = 200

reward = 0
done = False


def evaluate(agent_obs, nChildren):
	envObs = OpenAIGymEnv.from_spec({
		"type": "openai",
		"gym_env": 'gym_SmartPrimer:TestEnv-v0'
	})

	improvements = []
	for i in range(0, nChildren):
		ob_obs = envObs.reset()
		ob_obs = (ob_obs - [-4.5, 0, -5, 0, 0, 1.5, 25, 0]) / [9, 1, 10, 1, 1, 3, 50, 1]
		# ob_obs = (ob_obs - [4, 4, 0.5, 0.5, 0.5, 1.5, 15, 5]) / [8, 4, 1, 1, 1, 3, 30, 10]
		# action_list_obs = []

		while True:
			time_percentage_obs = min(agent_obs.timesteps / 1e6, 1.0)
			action = agent_obs.get_action(ob_obs, time_percentage=time_percentage_obs)
			# action = np.random.randint(0, 4)
			# action = 3

			# action_list_obs.append(action)

			next_ob_obs, reward, done, Baseinfo = envObs.step(action)
			next_ob_obs = (next_ob_obs - [-4.5, 0, -5, 0, 0, 1.5, 25, 0]) / [9, 1, 10, 1, 1, 3, 50, 1]
			# next_ob_obs = (next_ob_obs - [4, 4, 0.5, 0.5, 0.5, 1.5, 15, 5]) / [8, 4, 1, 1, 1, 3, 30, 10]

			# agent_obs.observe(ob_obs, action, None, reward, next_ob_obs, done)
			ob_obs = next_ob_obs

			if done:
				# print(envObs.gym_env.rewards)
				improvements.append(envObs.gym_env.rewards)

				agent_obs.reset()
				break

	return np.mean(improvements), np.std(improvements)


evaluation_improvements = []
for i in range(episode_count):
	print(i)
	# get the new children
	ob = env.reset()
	ob = (ob - [-4.5, 0, -5, 0, 0, 1.5, 25, 0]) / [9, 1, 10, 1, 1, 3, 50, 1]
	# ob = (ob - [-4.5, 0, 5, 0, 0, 0, 27, 0]) / [-9, 1, 10, 1, 1, 1, 50, 1]

	while True:
		time_percentage = min(agent.timesteps / 1e6, 1.0)
		action = agent.get_action(ob, time_percentage=time_percentage)

		next_ob, reward, done, Baseinfo = env.step(action)
		# next_ob = (next_ob - [4, 4, 0.5, 0.5, 0.5, 1.5, 15, 5]) / [8, 4, 1, 1, 1, 3, 30, 10]

		# print('observation: {}'.format(next_ob))
		# print('reward: {}'.format(reward))

		agent.observe(ob, action, None, reward, next_ob, done)
		next_ob = (next_ob - [-4.5, 0, -5, 0, 0, 1.5, 25, 0]) / [9, 1, 10, 1, 1, 3, 50, 1]
		ob = next_ob
		# next_ob = (next_ob - [-4.5, 0, 5, 0, 0, 0, 27, 0]) / [-9, 1, 10, 1, 1, 1, 50, 1]

		# if agent.timesteps % 200 == 0:
		# 	agent.update(time_percentage=time_percentage)

		# agent.update(time_percentage=time_percentage)
		# print('agent timesteps: {}'.format(agent.timesteps))
		if done:
			# print('Child is done')
			if i % 10 == 0:
				agent.update(time_percentage=time_percentage)

			if i % 10 == 0:
				agent_obs = copy.copy(agent) #copy the policy
				evaluation_improvement, evaluation_stds = evaluate(agent_obs, 300)

				evaluation_improvements.append(evaluation_improvement)

			agent.reset()
			break

# print(env.gym_env.info['Improvement'])
#
# print(evaluation_improvements)
plt.plot(evaluation_improvements)
plt.title('Improvement per agent update, averaged over 500 evaluation children')
plt.xlabel('Number of children trained x20')
plt.ylabel('Average improvement of 500 evaluation children')
plt.show()

print(env.gym_env.info['Performance'])
# make the plots
env.render()

performance = env.gym_env.info['Performance']
improvement = env.gym_env.info['Improvement']

pickle_name = '/Users/williamsteenbergen/Desktop/Smart_Primer/easy_PPO/per_ppo_psi03_' + str(args.seed) + '.pickle'
with open(pickle_name, 'wb') as handle:
	pickle.dump(performance, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle_name = '/Users/williamsteenbergen/Desktop/Smart_Primer/easy_PPO/imp_ppo_psi03_' + str(args.seed) + '.pickle'
with open(pickle_name, 'wb') as handle:
	pickle.dump(improvement, handle, protocol=pickle.HIGHEST_PROTOCOL)

actionInfo = env.gym_env.info['actionInfo']
pickle_name = '/Users/williamsteenbergen/Desktop/Smart_Primer/easy_PPO/actionInfo_ppo_psi03_' + str(
	args.seed) + '.pickle'
with open(pickle_name, 'wb') as handle:
	pickle.dump(actionInfo, handle, protocol=pickle.HIGHEST_PROTOCOL)
