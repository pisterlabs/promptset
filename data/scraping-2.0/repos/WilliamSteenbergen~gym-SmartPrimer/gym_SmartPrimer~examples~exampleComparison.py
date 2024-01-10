import gym
import numpy as np
import gym_SmartPrimer.agents.baselineV2 as Baseline
import matplotlib.pyplot as plt
from rlgraph.agents import Agent
from rlgraph.environments import OpenAIGymEnv
from rlgraph.execution import SingleThreadedWorker
import json
import os

np.random.seed(2)

env = gym.make('gym_SmartPrimer:SmartPrimer-realistic-v2')
agent = Baseline.BaselineAgent(env.action_space)

episode_count = 1000

reward = 0
done = False

for i in range(episode_count):
	if i == 100:
		stop = 1
	ob = env.reset()
	while True:
		action = agent.act(ob, reward, done)
		ob, reward, done, Baseinfo = env.step(action)
		if done:
			agent.reset()
			break


np.random.seed(2)

agent_config_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/agents/ppoSmartPrimer_config.json'

with open(agent_config_path, 'rt') as fp:
	agent_config = json.load(fp)

env = OpenAIGymEnv.from_spec({
        "type": "openai",
        "gym_env": 'gym_SmartPrimer:SmartPrimer-realistic-v2'
    })

agent = Agent.from_spec(
        agent_config,
        state_space=env.state_space,
        action_space=env.action_space
    )

episode_returns = []
def episode_finished_callback(episode_return, duration, timesteps, *args, **kwargs):
	episode_returns.append(episode_return)
	if len(episode_returns) % 100 == 0:
		print("Episode {} finished: reward={:.2f}, average reward={:.2f}.".format(
			len(episode_returns), episode_return, np.mean(episode_returns[-100:])
		))

worker = SingleThreadedWorker(env_spec=lambda: env, agent=agent, render=False, worker_executes_preprocessing=False,
                                  episode_finish_callback=episode_finished_callback)

# Use exploration is true for training, false for evaluation.
worker.execute_episodes(1000, use_exploration=True)

def plotting(Baseline, PPO, quit, finished, quitBaseline, finishedBaseline, actionInfo):
	ax1 = plt.subplot(311)
	#ax1.set_title('Scenario 4: average reward of last 100 children without quitting penalty')
	ax1.margins(0.05)
	#ax1.set_xlabel('Number of children')
	ax1.set_title('Average reward of last 100 children')
	ax1.plot(PPO, 'r', label='PPO')
	ax1.plot(Baseline, 'b', label='Baseline')
	ax1.legend()

	ax4 = plt.subplot(312)
	ax4.set_title('Actions taken')
	ax4.plot(actionInfo['question'], 'r', label='Question')
	ax4.plot(actionInfo['encourage'], 'g', label='Encourage')
	ax4.plot(actionInfo['hint'], 'b', label='Hint')
	ax4.plot(actionInfo['nothing'], 'y', label='Nothing')
	ax4.set_ylabel('% of last 500 actions')
	ax4.legend()

	ax2 = plt.subplot(325)
	ax2.set_title('PPO')
	ax2.plot(quit, 'r', label='Quit')
	ax2.plot(finished, 'g', label='Finished')
	ax2.set_ylabel('% of last 100 children')
	ax2.legend()

	ax3 = plt.subplot(326)
	ax3.set_title('Baseline')
	ax3.plot(quitBaseline, 'r', label='Quit')
	ax3.plot(finishedBaseline, 'g', label='Finished')

	ax3.legend()
	plt.show()


	ax1 = plt.subplot(141)
	ax1.set_title('Smart Child')
	ax1.plot(actionInfo['0'][0], 'y', label='Nothing')
	ax1.plot(actionInfo['0'][1], 'g', label='Encourage')
	ax1.plot(actionInfo['0'][2], 'r', label='Question')
	ax1.plot(actionInfo['0'][3], 'b', label='Hint')
	ax1.set_ylabel('% of last 500 actions')
	ax1.legend()

	ax2 = plt.subplot(142)
	ax2.set_title('Outgoing Child')
	ax2.plot(actionInfo['1'][0], 'y', label='Nothing')
	ax2.plot(actionInfo['1'][1], 'g', label='Encourage')
	ax2.plot(actionInfo['1'][2], 'r', label='Question')
	ax2.plot(actionInfo['1'][3], 'b', label='Hint')
	ax2.set_xlabel('Number of actions taken')
	ax2.legend()

	ax3 = plt.subplot(143)
	ax3.set_title('Quiet Child')
	ax3.plot(actionInfo['2'][0], 'y', label='Nothing')
	ax3.plot(actionInfo['2'][1], 'g', label='Encourage')
	ax3.plot(actionInfo['2'][2], 'r', label='Question')
	ax3.plot(actionInfo['2'][3], 'b', label='Hint')
	ax3.legend()

	ax4 = plt.subplot(144)
	ax4.set_title('Anxious Child')
	ax4.plot(actionInfo['3'][0], 'y', label='Nothing')
	ax4.plot(actionInfo['3'][1], 'g', label='Encourage')
	ax4.plot(actionInfo['3'][2], 'r', label='Question')
	ax4.plot(actionInfo['3'][3], 'b', label='Hint')
	ax4.legend()
	plt.show()

	return 0
plotting(Baseinfo['Performance'], env.gym_env.info['Performance'], env.gym_env.info['nQuit'], env.gym_env.info['nFinish'], Baseinfo['nQuit'], Baseinfo['nFinish'], env.gym_env.info['actionInfo'])

