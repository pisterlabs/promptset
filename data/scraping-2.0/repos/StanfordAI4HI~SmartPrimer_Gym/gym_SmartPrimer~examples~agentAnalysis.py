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
import time
import pandas as pd

def loadData(dirPath, weightCompPath):
	env = OpenAIGymEnv.from_spec({
					"type": "openai",
					"gym_env": 'gym_SmartPrimer:SmartPrimer-realistic-v2'
			})

	#configure the agent settings in this file
	agent_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),  'agents/ppoSmartPrimer_config.json')

	with open(agent_config_path, 'rt') as fp:
		agent_config = json.load(fp)

	fileNames = os.listdir(dirPath)
	fileNames.sort()

	results = np.loadtxt(weightCompPath)
	models = list(results[:,0])

	return env, agent_config, fileNames, models


def getActions(env, agent_config, fileNames, modelNumbers, listOfObs, repeats=1):
	int2action = ['hint', 'nothing', 'encourage', 'question']
	actionsDF = np.empty([len(listOfObs), len(modelNumbers)], dtype="U100")

	modelNumber = 0
	modelsUpdated = 0

	for modelName in fileNames:
		# if modelName == 'model-11-11-2020-18:39:02-PST-3.data-00000-of-00001':
		# 	print(modelNumber)
		# 	time.sleep(30)

		if modelName[-5:] == '00001':
			modelNumber += 1

			if modelNumber in modelNumbers:
				with open(dirPath + '/checkpoint', "w") as checkpointFile:
					print(modelName[0:-20])
					checkpointFile.write("model_checkpoint_path: \"" + modelName[0:-20] + "\" \n")
					checkpointFile.write("all_model_checkpoint_paths: \"" + modelName[0:-20] + "\" ")

				#retreive the agent from RLgraph
				agent = Agent.from_spec(
								agent_config,
								state_space=env.state_space,
								action_space=env.action_space,
						)

				agent.load_model(dirPath)

				for i in range(0, repeats):
					for actionNumber, obs in enumerate(listOfObs):
						action = agent.get_action(obs)
						actionsDF[actionNumber, modelsUpdated] += int2action[action] + ','

				modelsUpdated+=1
	actionsDF = np.array(actionsDF, dtype=str)
	return actionsDF


def getChangedModels(env, agent_config, fileNames):
	changedModels = []

	modelNumber = 0

	for modelName in fileNames:
		if modelName[-5:] == '00001':
			with open(dirPath + '/checkpoint', "w") as checkpointFile:
				checkpointFile.write("model_checkpoint_path: \"" + modelName[0:-20] + "\" \n")
				checkpointFile.write("all_model_checkpoint_paths: \"" + modelName[0:-20] + "\" ")

			modelNumber+=1
			print('Model number is: {}'.format(modelNumber))

			# retreive the agent from RLgraph
			# agent = Agent.from_spec(
			# 	agent_config,
			# 	state_space=env.state_space,
			# 	action_space=env.action_space,
			# )
			#
			# agent.load_model(dirPath)
			#
			# weigths = agent.get_weights()
			#
			# if modelNumber != 1:
			# 	checks = []
			# 	for key in weigths['policy_weights']:
			# 		if (weigths['policy_weights'][key] == prev_weights['policy_weights'][key]).all():
			# 			checks.append(True)
			# 		else:
			# 			checks.append(False)
			#
			# 	if not all(checks):
			# 		changedModels.append(modelNumber)
			# 		print('Changed model number is: {}'.format(modelNumber))
			#
			# prev_weights = copy.deepcopy(weigths)

	return changedModels
# resultFile.close()

def get_file_name(PT_number, model_name, fileNames):

	for j in range(PT_number, 33):
		model_name_PST = model_name + '-' + str(j) + '.data-00000-of-00001'
		model_name_PST_list = [i for i in fileNames if model_name_PST in i]



		if len(model_name_PST_list) != 0:
			index_new_PST = fileNames.index(model_name_PST)

			model_name_return = fileNames[index_new_PST-6]

			PT_number = j
			prev_model = model_name
			break

	return  model_name_return[:-len('.data-00000-of-00001')], PT_number, prev_model

def get_action_probs(env, agent_config, dataFile, fileNames, repeats=1000):
	data = []

	actions_df_with_times = pd.read_csv(data_file_times)[8:]

	prev_model = ''
	PT_number = 0
	prev_uid = 0

	first_time = True

	model_round = -1
	for actionNumber, action in actions_df_with_times.iterrows():
		model_name = 'model-' + action['time_stored'].replace('\n', '')
		if model_name == prev_model and action['user_id'] != prev_uid:
			prev_uid = action['user_id']
			PT_number+=1
			model_name_PT, PT_number, prev_model = get_file_name(PT_number, model_name, fileNames)

		else:
			PT_number = 0
			prev_uid = action['user_id']
			model_name_PT, PT_number, prev_model = get_file_name(PT_number, model_name, fileNames)


		with open(dirPath + '/checkpoint', "w") as checkpointFile:

			# model_name = 'model-10-31-2020-16:03:36-PDT'
			checkpointFile.write("model_checkpoint_path: \"" + model_name_PT + "\" \n")
			checkpointFile.write("all_model_checkpoint_paths: \"" + model_name_PT + "\" ")

		# retreive the agent from RLgraph
		agent = Agent.from_spec(
			agent_config,
			state_space=env.state_space,
			action_space=env.action_space,
		)

		agent.load_model(dirPath)

		weigths = agent.get_weights()

		checks = []

		for key in weigths['policy_weights']:
			if not first_time and (weigths['policy_weights'][key] == prev_weights['policy_weights'][key]).all():
				checks.append(True)
			else:
				checks.append(False)

		first_time = False
		if not all(checks):
			model_round +=1

		# obs = [action['grade_norm'], action['pre-score_norm'], action['stage_norm'], action['failed_attempts_norm'], action['pos_norm'], action['neg_norm'], action['hel_norm'], action['anxiety_norm']]
		# frequency = [0,0,0,0]
		# for i in range(0, repeats):
		# 	action = agent.get_action(obs)
		# 	frequency[action] += 1

		# data.append([actionNumber] + [x / repeats for x in frequency] + [model_round])
		data.append([actionNumber, model_round])

		prev_weights = copy.deepcopy(weigths)

	# data = pd.DataFrame(data, columns=['action_number', 'p_hint', 'p_nothing', 'p_encourage', 'p_question', 'model_round'])
	# data = data.set_index(data['action_number'])

	data = pd.DataFrame(data, columns=['action_number', 'model_round'])
	data = data.set_index(data['action_number'])

	final_data = pd.concat([actions_df_with_times, data], axis=1)
	return final_data

if __name__ == '__main__':
	dirPath = "/Users/williamsteenbergen/Documents/Stanford/SmartPrimer/Final_Logs/store"
	weightCompPath = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/gym_SmartPrimer/examples/weightComparisons.txt'

	listOfObs = [
		[-1., -0.5,-0.33333333,-0.7,-1.,-1,1,-0.83333333],
		[-1,-0.5,0,-0.8, -1,-1,1,-0.833333],
		[-1,-0.5,0,-0.8, -1,0.076,-1,-0.833333],
		[-1,-0.5,0,-0.4, -1,-1,-1,-0.833333],
		[-1,-0.5,0,-0.4, -0.032, -0.29,-1,-0.8333333],
		[-1,-0.5,0, 0.2, -1, 0.334,-1,-0.833333],
		[-1,-0.5,0,0.2,1,-1,1,-0.8333333],
		[-1,-0.5,0,0.3,1,-1,1,-0.8333333]
	]

	listOfObs = [
		[0, -0.25, -1, -1, -1, -1, 1, -0.16666667],
		[0, -0.25, -1, -1, -1, -1, 1, -0.16666667],
		[0, -0.25, -0.33333, -1, -1, -1, 1, -0.1666667],
		[0, -0.25, -0.333333, -0.8, 1, -1, 1, -0.1666667],
		[0, -0.25, 0, -0.9, -1, -1, 1, -0.1666667],
		[0, -0.25, 0.6666667, -1, 1, -1, -0.1063473, -0.16777777]
	]

	env, agent_config, fileNames, models = loadData(dirPath, weightCompPath)

	# changedModels = getChangedModels(env, agent_config, fileNames)

	# modelNumbers = [65]
	#
	# data = pd.read_csv('/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/gym_SmartPrimer/examples/offline_with_probabilities.csv')
	#
	# hint_mean = np.mean(data['p_hint'])
	# encourage_mean = np.mean(data['p_encourage'])
	# question_mean = np.mean(data['p_question'])
	# nothing_mean = np.mean(data['p_nothing'])
	#
	# print("hint mean: {}, encourage mean: {}, question mean: {}, nothing mean: {}".format(hint_mean, encourage_mean, question_mean, nothing_mean))



	offline_data_file = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/ExperimentAnalysis/Final_analysis/offline_rl_data.csv'
	data_file_times = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/ExperimentAnalysis/Final_analysis/offline_rl_data_with_times.csv'
	actionData = get_action_probs(env, agent_config, offline_data_file, fileNames, repeats=1000)

	actionData.to_csv('offline_with_probabilities_and_model_round.csv')


	#
	# actionsDF = getActions(env, agent_config, fileNames, modelNumbers, listOfObs, repeats=5)
	#
	# np.savetxt('model77_88_child215.csv', actionsDF, delimiter=',', fmt='%s')

# print("ModelNumber is a changing model: {}".format(modelNumber))
# distance = 0
# for key in weigths['policy_weights']:
# 	distance += np.sum((weigths['policy_weights'][key] - prev_weights['policy_weights'][key]) ** 2)
#
# distance = np.sqrt(distance)






