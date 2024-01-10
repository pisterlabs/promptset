import gym
from gym import wrappers
import numpy as np
from rlgraph.agents import Agent
from rlgraph.environments import OpenAIGymEnv
import json
import os
import copy
import pandas as pd
import time

def loadData(dirPath):
	env = OpenAIGymEnv.from_spec({
					"type": "openai",
					"gym_env": 'gym_SmartPrimer:SmartPrimer-realistic-v2'
			})

	#configure the agent settings in this file
	agent_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),  'OfflineRL/ppoSmartPrimer_config.json')

	with open(agent_config_path, 'rt') as fp:
		agent_config = json.load(fp)

	fileNames = os.listdir(dirPath)
	fileNames.sort()

	return env, agent_config, fileNames

def get_action_probs(env, agent_config, repeats=1000):
	data = []

	actions_df_with_times = pd.read_csv('cleaned_data.csv')
	actions_df_with_times = actions_df_with_times.sort_values('time_stamp')

	actions_df_with_times = actions_df_with_times.reset_index()

	model_round = 0
	rounds = []
	first = True
	for action_number, row in actions_df_with_times.iterrows():
		model_name = row['model_name']

		with open(dirPath + '/checkpoint', "w") as checkpointFile:
			print(model_name)
			print("--------------------------------------------------------")
			checkpointFile.write("model_checkpoint_path: \"" + model_name[0:-5] + "\" \n")
			checkpointFile.write("all_model_checkpoint_paths: \"" + model_name[0:-5] + "\" ")
			time.sleep(1)
		checkpointFile.close()

		# retreive the agent from RLgraph
		agent = Agent.from_spec(
			agent_config,
			state_space=env.state_space,
			action_space=env.action_space,
		)

		agent.load_model(dirPath)

		weigths = agent.get_weights()

		no_match = True
		for round_number, other_round in enumerate(rounds):
			checks = []

			for key in weigths['policy_weights']:
				if  (weigths['policy_weights'][key] == other_round['policy_weights'][key]).all():
					checks.append(True)
				else:
					checks.append(False)

			if all(checks):
				model_round = round_number
				no_match = False
				break

		if first:
			rounds.append(weigths)
			first = False

		elif no_match:
			rounds.append(weigths)
			model_round = len(rounds) - 1

		obs = [row['grade'], row['pre_score'], row['stage'], row['failed_attempts'], row['pos'], row['neg'], row['help'], row['anxiety']]
		frequency = [0,0,0,0]
		for i in range(0, repeats):
			action = agent.get_action(obs)
			frequency[action] += 1

		data.append([x / repeats for x in frequency] + [model_round])

	data = pd.DataFrame(data, columns=['p_hint', 'p_nothing', 'p_encourage', 'p_question', 'model_round'])
	data = data.reset_index()

	final_data = pd.concat([actions_df_with_times, data], axis=1)
	return final_data

if __name__ == '__main__':
	dirPath = "/Users/williamsteenbergen/Documents/Stanford/SmartPrimer/Final_Logs/store"

	env, agent_config, fileNames = loadData(dirPath)

	result = get_action_probs(env, agent_config, repeats=1000)
	result.to_csv('new_data.csv')