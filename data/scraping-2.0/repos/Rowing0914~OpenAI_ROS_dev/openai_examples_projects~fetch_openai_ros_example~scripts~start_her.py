#! /usr/bin/env python
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import rospy
import tensorflow as tf
from datetime import datetime
from collections import deque
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import eager_setup
from tf_rl.common.params import ROBOTICS_ENV_LIST
from tf_rl.common.train import train_HER
from tf_rl.common.networks import HER_Actor as Actor, HER_Critic as Critic
from tf_rl.agents.DDPG import HER_DDPG as DDPG

eager_setup()

"""
# defined in params.py
ROBOTICS_ENV_LIST = {
    "FetchPickAndPlace-v1": 0,
    "FetchPush-v1": 0,
    "FetchReach-v1": 0,
    "FetchSlide-v1": 0
}
"""


class P:
	def __init__(self):
		self.env_name = "FetchReach-v1"
		self.seed = 123
		# number of epochs for training
		self.num_epochs = 200
		# number of cycles in an epoch
		self.num_cycles = 50
		# number of episodes in a cycle
		self.num_episodes = 16
		# number of replay strategy
		self.replay_k = 4
		# number of updates in a cycle
		self.num_updates = 40
		self.memory_size = 100000
		self.batch_size = 256
		self.soft_update_tau = 0.05
		self.gamma = 0.98
		self.action_l2 = 1.0
		self.noise_eps = 0.2
		self.random_eps = 0.3
		self.log_dir = "./logs/logs"
		self.model_dir = "./logs/models"


params = P()

params.goal = ROBOTICS_ENV_LIST[params.env_name]
params.test_episodes = 10

now = datetime.now()

params.log_dir += "{}".format(params.env_name)
params.model_dir += "{}".format(params.env_name)

rospy.init_node("start_her")
task_and_robot_environment_name = rospy.get_param(
	'/fetch/task_and_robot_environment_name')
# to register our task env to openai env.
# so that we don't care the output of this method for now.
env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

# params.max_action = env.action_space.high[0]
# params.num_action = env.action_space.shape[0]

# TODO: this is temp solution..... check openai's fetch's implementation!!
params.max_action = 0
params.num_action = 4

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)

agent = DDPG(Actor, Critic, params.num_action, params)

replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.num_episodes)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)
train_HER(agent, env, replay_buffer, reward_buffer, summary_writer)
