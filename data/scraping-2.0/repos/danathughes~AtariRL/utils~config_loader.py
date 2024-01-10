## config_loader.py			Dana Hughes				21-Sept-2017
##
## Utility for loading a configuration from a config file


# Environments
from environments.AtariEnvironment import AtariEnvironment
from environments.OpenAIGymEnvironment import OpenAIGymEnvironment

# Networks
from models.networks import *
#from utils.builders.network_builders import *

# Memory
from memory.memory import ReplayMemory
from memory.priority_memory import PriorityReplayMemory, RankedPriorityReplayMemory
from memory.bootstrapped_memory import BootstrappedReplayMemory

# Agents
from agents.dqn_agent import DQN_Agent, DoubleDQN_Agent
from agents.bootstrapped_dqn_agent import Bootstrapped_DQN_Agent
from agents.epsilon_agent import EpsilonAgent

from listeners.checkpoint_recorder import CheckpointRecorder
from listeners.tensorboard_monitor import TensorboardMonitor

from utils.counter import Counter


import ConfigParser


def load_str(config, section, name, default):
	"""
	Load a string from the configuration 
	"""

	try:
		value = config.get(section, name)
	except:
		value = default

	return value


def load_int(config, section, name, default):
	"""
	Load an integer from the configuration
	"""

	try:
		value = config.getint(section, name)
	except:
		value = default

	return value


def load_float(config, section, name, default):
	"""
	Load a float from the configuration
	"""

	try:
		value = config.getfloat(section, name)
	except:
		value = default

	return value


def load(config_filename):
	"""
	Loads a configuration of compoents from a file

	config_filename - Name of the file to load components from
	"""

	# Does the filename exist?

	# Load the configuration file
	config = ConfigParser.RawConfigParser()
	config.read(config_filename)

	# Parse and construct each part
	environment = load_environment(config)
	dqn, target_dqn = load_network(config, environment)
	memory = load_memory(config, environment)

	# Make a counter
	counter_start = load_int(config, 'Counter', 'start', 0)
	counter = Counter(counter_start)

	dqn_agent, agent, eval_agent = load_agent(config, environment, dqn, target_dqn, memory, counter)

	# Create a checkpoint object to save the agent and memory
	checkpoint = load_checkpoint(config, dqn_agent.dqn, memory, counter)
	tensorboard = load_tensorboard(config, dqn_agent, counter)

	return environment, agent, eval_agent, counter, checkpoint, tensorboard



def load_agent(config, environment, dqn, target_dqn, memory, counter):
	"""
	Load an agent from the ConfigParser
	"""

	# Which type of base agent is this?  Default to DQN
	agent_type = load_str(config, 'Agent', 'type', 'DQN')

	# Get all the agent parameters
	replay_start_size = load_int(config, 'Agent', 'replay_start_size', 50000)
	target_update_frequency = load_int(config, 'Agent', 'target_update_frequency', 10000)
	update_frequency = load_int(config, 'Agent', 'update_frequency', 4)
	minibatch_size = load_int(config, 'Agent', 'minibatch_size', 32)
	discount_factor = load_float(config, 'Agent','discount_factor', 0.99)
	history_size = load_int(config, 'Agent', 'history_size', 4)

	frame_shape = environment.screen_size
	num_actions = environment.num_actions

	# Build the agent!
	if agent_type == "DQN":
		dqn_agent = DQN_Agent(frame_shape, num_actions, history_size, dqn, target_dqn, memory,
									 minibatch_size=minibatch_size, discount_factor=discount_factor)
	elif agent_type == "DoubleDQN":
		dqn_agent = DoubleDQN_Agent(frame_shape, num_actions, history_size, dqn, target_dqn, memory,
									 minibatch_size=minibatch_size, discount_factor=discount_factor)
	elif agent_type == "BootstrappedDQN":
		num_heads = load_int(config, 'Network', 'num_heads', 10)
		dqn_agent = Bootstrapped_DQN_Agent(frame_shape, num_actions, history_size, dqn, target_dqn, memory, num_heads,
									 minibatch_size=minibatch_size, discount_factor=discount_factor)

	# Add callbacks to the counter for the dqn agent
	counter.add_hook(dqn_agent.update_target_network, target_update_frequency, 0)
	counter.add_hook(dqn_agent.train, update_frequency, replay_start_size)


	# Create epsilon agents
	initial_epsilon = load_float(config, 'Agent', 'initial_epsilon', 1.0)
	final_epsilon = load_float(config, 'Agent', 'final_epsilon', 0.1)
	initial_frame = load_int(config, 'Agent', 'initial_epsilon_frame', 0)
	final_frame = load_int(config, 'Agent', 'final_epsilon_frame', 1000000)
	eval_epsilon = load_int(config, 'Agent', 'evaluate_epsilon', 0.05)

	# Make two agents -- a training epsilon agent, and an evaluation agent
	agent = EpsilonAgent(dqn_agent, counter, initial_epsilon, final_epsilon, initial_frame, final_frame)
	eval_agent = EpsilonAgent(dqn_agent, counter, eval_epsilon, eval_epsilon, 1, 1)

	return dqn_agent, agent, eval_agent


def load_environment(config):
	"""
	Load an environment from the ConfigParser
	"""

	# Try to get values from the config file

	# Which game to load
	try:
		game_path = config.get('Environment', 'game_path')
	except:
		print "game_path not defined in Environment!"
		return None

	# Which environment class should be used?  Defaults to AtariEnvironment
	env_class = load_str(config, 'Environment', 'class', 'AtariEnvironment')

	# Get the scaled screen dimensions - default is (84,84)
	width = load_int(config, 'Environment', 'width', 84)
	height = load_int(config, 'Environment', 'height', 84)
	
	# Build the environment
	if env_class == "AtariEnvironment":
		return AtariEnvironment(game_path, screen_size=(width, height))
	elif env_class == "OpenAIGym":
		return OpenAIGymEnvironment(game_path)
	else:
		print "Unknown environment class: %s" % env_class
		return None


def load_memory(config, environment):
	"""
	Load memory from the ConfigParser
	"""

	# Which type of memory to use?  Default to ReplayMemory
	memory_type = load_str(config, 'Memory', 'type', 'ReplayMemory')
	base_memory_type = load_str(config, 'Memory', 'base_type', 'ReplayMemory')
	size = load_int(config, 'Memory', 'size', 1000000)
	alpha = load_float(config, 'Memory', 'alpha', 0.6)
	beta = load_float(config, 'Memory', 'beta', 0.4)
	epsilon = load_float(config, 'Memory', 'epsilon', 1e-6)
	mask_function = load_str(config, 'Memory', 'mask_function', 'binomial')

	# Create the memory
	if memory_type == "ReplayMemory":
		memory = ReplayMemory(size, environment.screen_size)

	elif memory_type == "PriorityReplayMemory":
		memory = PriorityReplayMemory(size, environment.screen_size, alpha, beta, epsilon)

	elif memory_type == "RankedPriorityReplayMemory":
		memory = RankedPriorityReplayMemory(size, environment.screen_size, alpha, beta)

	elif memory_type == "BootstrappedReplayMemory":

		# Create the base memory, default to ReplayMemory
		if base_memory_type == "PriorityReplayMemory":
			base_memory = PriorityReplayMemory(size, environment.screen_size, alpha, beta, epsilon)
		elif base_memory_type == "RankedPriorityReplayMemory":
			base_memory = RankedPriorityReplayMemory(size, environment.screen_size, alpha, beta)
		else:
			base_memory = ReplayMemory(size, environment.screen_size)
	
		num_heads = load_int(config, 'Network', 'num_heads', 10)
		memory = BootstrappedReplayMemory(size, base_memory, num_heads)

	else:
		print "Unknown memory type: %s" % memory_type
		return None

	return memory


def load_network(config, environment):
	"""
	Load a neural network builder
	"""

	# Which type of network agent is this?  Default to DQN
	network_type = load_str(config, 'Network', 'type', 'DQN')

	# Which architecture should be used?  Default to NATURE
	architecture = load_str(config, 'Network', 'architecture', 'NATURE')

	# Bootstrapped DQN requires knowledge of the number of heads.
	# Default to 10
	num_heads = load_int(config, 'Network', 'num_heads', 10)

	# Make a DQN and Target DQN
	history_size = load_int(config, 'Agent', 'history_size', 4)
	input_shape = environment.screen_size + (history_size, )
	num_actions = environment.num_actions

	# Use the appropriate architecture
	if architecture == "NIPS":
		dqn = nips_dqn(input_shape, num_actions, network_name='dqn')
		target_dqn = nips_dqn(input_shape, num_actions, network_name='target_dqn', trainable=False)
	elif architecture == "NATURE":
		dqn = nature_dqn(input_shape, num_actions, network_name='dqn')
		target_dqn = nature_dqn(input_shape, num_actions, network_name='target_dqn', trainable=False)
	elif architecture == "DUELING":
		dqn = deuling_dqn(input_shape, num_actions, network_name='dqn')
		target_dqn = deuling_dqn(input_shape, num_actions, network_name='target_dqn', trainable=False)
	else:
		print "Unknown network architecture: %s" % architecture
		return None, None

	return dqn, target_dqn


def load_checkpoint(config, dqn, memory, counter):
	"""
	Load checkpoint object
	"""

	# What's the base path to save to?
	checkpoint_path = load_str(config, 'Checkpoint', 'path', './checkpoint')

	# How often to save the DQN parameters, tensorflow graph and memory?
	dqn_save_rate = load_int(config, 'Checkpoint', 'dqn_save_rate', 100000)
	tensorflow_save_rate = load_int(config, 'Checkpoint', 'tensorflow_save_rate', 100000)
	memory_save_rate = load_int(config, 'Checkpoint', 'memory_save_rate', 1000000)

	checkpoint = CheckpointRecorder(dqn, memory, counter, checkpoint_path)

	# Add the hooks to the counter
	counter.add_hook(checkpoint.save_dqn, dqn_save_rate)
	counter.add_hook(checkpoint.save_memory, memory_save_rate)
	counter.add_hook(checkpoint.save_tensorflow, tensorflow_save_rate)

	return checkpoint


def load_tensorboard(config, dqn_agent, counter):
	"""
	Create a Tensorboard monitoring object
	"""

	# What's the path for logging?
	log_path = load_str(config, 'Tensorboard', 'path', './log')

	tensorboard = TensorboardMonitor(log_path, counter)
	tensorboard.add_scalar_summary('score', 'per_game_summary')
	tensorboard.add_scalar_summary('training_loss', 'training_summary')
	for i in range(dqn_agent.num_actions):
		tensorboard.add_histogram_summary('Q%d_training' % i, 'training_summary')

	# Allow the dqn agent access to the tensorboard monitor to report statistics
	dqn_agent.add_listener(tensorboard)

	return tensorboard