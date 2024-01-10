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
import csv
from tensorflow import keras


def evaluate(agent_obs, nChildren, model):
	envObs = OpenAIGymEnv.from_spec({
		"type": "openai",
		"gym_env": 'gym_SmartPrimer:SmartPrimer-realistic-v2'
	})

	improvements = []
	for i in range(0, nChildren):
		np.random.seed(i)
		ob_obs = envObs.reset()
		ob_obs = (ob_obs - [4, 4, 0.5, 0.5, 0.5, 1.5, 15, 5]) / [8, 4, 1, 1, 1, 3, 30, 10]
		action_list_obs = []
		# print(ob_obs)

		while True:
			time_percentage_obs = min(agent_obs.timesteps / 1e6, 1.0)
			action = agent_obs.get_action(ob_obs, time_percentage=time_percentage_obs)
			# action = np.random.randint(0, 4)
			# action = 3
			# action = 0
			# input = np.array(ob_obs)
			# input = np.expand_dims(input, axis=0)
			# action = model.predict(input)
			# action = np.argmax(action)
			# print(action)

			action_list_obs.append(action)

			next_ob_obs, reward, done, Baseinfo = envObs.step(action)
			next_ob_obs = (next_ob_obs - [4, 4, 0.5, 0.5, 0.5, 1.5, 15, 5]) / [8, 4, 1, 1, 1, 3, 30, 10]

			ob_obs = next_ob_obs

			if done:
				improvements.append(envObs.gym_env.info['improvementPerChild'])

				agent_obs.reset()
				break

	return improvements, np.std(improvements)

def runSimulation(seed, steps, stepsupdate, model):
	results = []

	np.random.seed(seed)
	# create the environment
	env = OpenAIGymEnv.from_spec({
		"type": "openai",
		"gym_env": 'gym_SmartPrimer:SmartPrimer-realistic-v2'
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
	episode_count = steps

	evaluation_improvements = []


	actions = []
	observations = []
	actionToInt = {'hint': 0,
	               'question': 1,
	               'encourage': 2,
	               'nothing': 3
	               }
	for i in range(episode_count):
		# get the new children
		ob = env.reset()
		ob = (ob - [4, 4, 0.5, 0.5, 0.5, 1.5, 15, 5]) / [8, 4, 1, 1, 1, 3, 30, 10]

		while True:
			observations.append(ob)
			time_percentage = min(agent.timesteps / 1e6, 1.0)
			action = agent.get_action(ob, time_percentage=time_percentage)

			next_ob, reward, done, Baseinfo = env.step(action)
			next_ob = (next_ob - [4, 4, 0.5, 0.5, 0.5, 1.5, 15, 5]) / [8, 4, 1, 1, 1, 3, 30, 10]
			actions.append(actionToInt[Baseinfo['action']])

			# print('observation: {}'.format(next_ob))
			# print('reward: {}'.format(reward))

			agent.observe(ob, action, None, reward, next_ob, done)
			ob = next_ob

			# if agent.timesteps % 200 == 0:
			# 	agent.update(time_percentage=time_percentage)

			# agent.update(time_percentage=time_percentage)
			# print('agent timesteps: {}'.format(agent.timesteps))
			if done:

				# print('Child is done')
				if i % stepsupdate == 0 or i == steps-1:
					agent.update(time_percentage=time_percentage)

					if i % (steps-1) == 0 and i != 0:
						agent_obs = copy.copy(agent)  # copy the policy
						results, evaluation_stds = evaluate(agent_obs, 1000, model)
						evaluation_improvements = evaluation_improvements + results
						agent.reset()
						break

				agent.reset()
				break

	return(evaluation_improvements)

improvements = [0]

model = keras.models.load_model('/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/behave_cloning6.h5')
for i in range(0, 10):
	result = runSimulation(i, steps=2000, stepsupdate=10, model=model)
	improvements = improvements + result

np.savetxt("behaveCloning20.csv", np.array(improvements)[1:], delimiter=",")



