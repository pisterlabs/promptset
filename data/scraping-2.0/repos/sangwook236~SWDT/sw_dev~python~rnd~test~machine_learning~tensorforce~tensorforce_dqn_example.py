# REF [site] >> http://tensorforce.readthedocs.io/en/latest/runner.html

# Path to libcudnn.so.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#--------------------
import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
	lib_home_dir_path = '/home/sangwook/lib_repo/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
	#lib_home_dir_path = 'D:/lib_repo/python'
	lib_home_dir_path = 'D:/lib_repo/python/rnd'
sys.path.append(swl_python_home_dir_path + '/src')
sys.path.append(lib_home_dir_path + '/tensorforce_github')
sys.path.append(lib_home_dir_path + '/gym_github')

#---------------------------------------------------------------------

import logging

from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner

gym_id = 'CartPole-v0'
batch_size = 64
num_episodes = 10000
num_episode_timesteps = 1000
report_episodes = 10

env = OpenAIGym(gym_id)
network_spec = [
	dict(type='dense', size=32, activation='tanh'),
	dict(type='dense', size=32, activation='tanh')
]

agent = DQNAgent(
	states_spec=env.states,
	actions_spec=env.actions,
	network_spec=network_spec,
	batch_size=batch_size
)

runner = Runner(agent=agent, environment=env)

def episode_finished(r):
	if r.episode % report_episodes == 0:
		logging.info('Finished episode {ep} after {ts} timesteps'.format(ep=r.episode, ts=r.timestep))
		logging.info('Episode reward: {}'.format(r.episode_rewards[-1]))
		logging.info('Average of last 100 rewards: {}'.format(sum(r.episode_rewards[-100:]) / 100))
	return True

print('Starting {agent} for Environment '{env}''.format(agent=agent, env=env))

runner.run(episodes=num_episodes, max_episode_timesteps=num_episode_timesteps, episode_finished=episode_finished)

print('Learning finished. Total episodes: {ep}'.format(ep=runner.episode))
