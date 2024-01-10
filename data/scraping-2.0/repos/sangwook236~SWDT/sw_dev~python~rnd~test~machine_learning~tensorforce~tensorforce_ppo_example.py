# REF [site] >> https://github.com/reinforceio/tensorforce
# REF [site] >> http://tensorforce.readthedocs.io/en/latest/

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

import numpy as np

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

batch_size = 4096
num_episodes = 3000
num_episode_timesteps = 200

# Create an OpenAIgym environment.
#env = OpenAIGym('CartPole-v0', visualize=True)
env = OpenAIGym('CartPole-v0')

# Network as list of layers.
network_spec = [
	dict(type='dense', size=32, activation='tanh'),
	dict(type='dense', size=32, activation='tanh')
]

agent = PPOAgent(
	states_spec=env.states,
	actions_spec=env.actions,
	network_spec=network_spec,
	batch_size=batch_size,
	# BatchAgent.
	keep_last_timestep=True,
	# PPOAgent.
	step_optimizer=dict(
		type='adam',
		learning_rate=1e-3
	),
	optimization_steps=10,
	# Model.
	scope='ppo',
	discount=0.99,
	# DistributionModel.
	distributions_spec=None,
	entropy_regularization=0.01,
	# PGModel.
	baseline_mode=None,
	baseline=None,
	baseline_optimizer=None,
	gae_lambda=None,
	# PGLRModel.
	likelihood_ratio_clipping=0.2,
	summary_spec=None,
	distributed_spec=None
)

# Create the runner.
runner = Runner(agent=agent, environment=env)

# Callback function printing episode statistics
def episode_finished(r):
	print('Finished episode {ep} after {ts} timesteps (reward: {reward})'.format(ep=r.episode, ts=r.episode_timestep, reward=r.episode_rewards[-1]))
	return True

# Start learning.
runner.run(episodes=num_episodes, max_episode_timesteps=num_episode_timesteps, episode_finished=episode_finished)

# Print statistics.
print('Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.'.format(
	ep=runner.episode,
	ar=np.mean(runner.episode_rewards[-100:]))
)
