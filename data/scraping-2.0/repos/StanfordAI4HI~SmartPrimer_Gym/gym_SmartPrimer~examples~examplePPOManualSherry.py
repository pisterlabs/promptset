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


parser = argparse.ArgumentParser(description='example')
parser.add_argument('--seed', type=int, default=0,
                    help='numpy seed ')
parser.add_argument('--time', type=int, default=3,
                    help='numpy seed ')
args = parser.parse_args()

np.random.seed(args.seed)

#create the environment
env = OpenAIGymEnv.from_spec({
				"type": "openai",
				"gym_env": 'gym_SmartPrimer:SmartPrimer-realistic-v2'
		})


#configure the agent settings in this file
agent_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),  'agents/ppoSmartPrimer_config.json')

with open(agent_config_path, 'rt') as fp:
	agent_config = json.load(fp)

#retreive the agent from RLgraph
agent = Agent.from_spec(
				agent_config,
				state_space=env.state_space,
				action_space=env.action_space
		)

#define number of children to simulate
episode_count = 101

reward = 0
done = False

def evaluate(agent_obs, nChildren):
	observations = []
	actions = []
	envObs = OpenAIGymEnv.from_spec({
		"type": "openai",
		"gym_env": 'gym_SmartPrimer:SmartPrimer-realistic-v2'
	})
	convert=['nothing', 'encourage', 'question',  'hint']
	improvements = []
	for i in range(0, nChildren):
		np.random.seed(i)
		ob_obs = envObs.reset()
		ob_obs = (ob_obs - [4, 4, 0.5, 0.5, 0.5, 1.5, 15, 5]) / [8, 4, 1, 1, 1, 3, 30, 10]
		action_list_obs = []
		# print(ob_obs)

		while True:
			time_percentage_obs = min(agent_obs.timesteps / 1e6, 1.0)
			# action = agent_obs.get_action(ob_obs, time_percentage=time_percentage_obs)
			action = np.random.randint(0, 4)
			# action = 3

			# action_list_obs.append(convert.index(action))
			observations.append(ob_obs)

			next_ob_obs, reward, done, Baseinfo = envObs.step(action)
			next_ob_obs = (next_ob_obs - [4, 4, 0.5, 0.5, 0.5, 1.5, 15, 5]) / [8, 4, 1, 1, 1, 3, 30, 10]
			actions.append(convert.index(Baseinfo['action']))

			ob_obs = next_ob_obs

			if done:
				improvements.append(envObs.gym_env.info['improvementPerChild'])

				agent_obs.reset()
				break

	# np.savetxt("onlyHints.csv", np.array(improvements), delimiter=",")
	return np.mean(improvements), np.std(improvements), observations, actions

evaluation_improvements = []

actions = []
observations = []
actionToInt = {'hint': 0,
               'question': 1,
               'encourage': 2,
               'nothing': 3
}
for i in range(episode_count):
		#get the new children
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
						if i % 10 == 0:
							agent.update(time_percentage=time_percentage)

							if i % 100 == 0 and i!=0:
								agent_obs = copy.copy(agent) #copy the policy
								evaluation_improvement, evaluation_stds, obFinal, actFinal = evaluate(agent_obs, 10000)
								# print(evaluation_improvement)
								# print(evaluation_stds)
								evaluation_improvements.append(evaluation_improvement)


						agent.reset()
						break

# print(env.gym_env.info['Improvement'])
#
# print(evaluation_improvements)

# with open('labels3.csv', 'w') as labelFile:
# 	wr = csv.writer(labelFile, quoting=csv.QUOTE_ALL)
# 	wr.writerow(actions)

np.savetxt('labels6.csv', np.array(actFinal, dtype=np.int), delimiter=',')

observations = np.array(observations)
np.savetxt("observations6.csv", np.array(obFinal, dtype=np.float), delimiter=",")

plt.plot(evaluation_improvements)
plt.title('Improvement per agent update, averaged over 300 evaluation children')
plt.xlabel('Number of children trained x20')
plt.ylabel('Average improvement of 300 evaluation children')
plt.show()

print(env.gym_env.info['Performance'])
# make the plots
env.render()

performance = env.gym_env.info['Performance']
improvement = env.gym_env.info['Improvement']
                       
pickle_name = '/Users/williamsteenbergen/Desktop/Smart_Primer/easy_PPO/per_ppo_psi03_'+str(args.seed)+'.pickle'
with open(pickle_name , 'wb') as handle:
    pickle.dump(performance, handle, protocol=pickle.HIGHEST_PROTOCOL)
pickle_name = '/Users/williamsteenbergen/Desktop/Smart_Primer/easy_PPO/imp_ppo_psi03_'+str(args.seed)+'.pickle'
with open(pickle_name , 'wb') as handle:
    pickle.dump(improvement, handle, protocol=pickle.HIGHEST_PROTOCOL)

actionInfo = env.gym_env.info['actionInfo']    
pickle_name = '/Users/williamsteenbergen/Desktop/Smart_Primer/easy_PPO/actionInfo_ppo_psi03_'+str(args.seed)+'.pickle'
with open(pickle_name , 'wb') as handle:
    pickle.dump(actionInfo, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    