# # FAPS PLMAgents
import logging
import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras import backend as k

from OpenAIGym.exception import FAPSPLMEnvironmentException

logger = logging.getLogger("FAPSPLMAgents")


class FAPSTrainerException(FAPSPLMEnvironmentException):
    """
    Related to errors with the Trainer.
    """
    pass


class SARSA:
    """This class is the abstract class for the trainers"""

    def __init__(self, envs, brain_name, trainer_parameters, training, seed):
        """
        Responsible for collecting experiences and training a neural network model.

        :param envs: The FAPSPLMEnvironment.
        :param brain_name: The brain to train.
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param seed: Random seed.
        """

        # initialize global trainer parameters
        self.brain_name = brain_name
        self.trainer_parameters = trainer_parameters
        self.is_training = training
        self.seed = seed
        self.steps = 0
        self.last_reward = 0
        self.initialized = False

        # initialize specific DQN parameters
        self.env_brains = envs
        self.state_size = 0
        self.action_size = 0
        for k, env in self.env_brains.items():
            self.action_size = env.action_space.n
            self.state_size = env.observation_space.n

        self.batch_size = self.trainer_parameters['batch_size']
        self.gamma = self.trainer_parameters['gamma']  # discount rate
        self.alpha = self.trainer_parameters['alpha']
        self.alpha_decay = self.trainer_parameters['alpha_decay']
        self.alpha_min = self.trainer_parameters['alpha_min']
        self.epsilon = self.trainer_parameters['epsilon']  # exploration rate
        self.epsilon_min = self.trainer_parameters['epsilon_min']
        self.epsilon_decay = self.trainer_parameters['epsilon_decay']
        self.replay_memory = deque(maxlen=self.trainer_parameters['memory_size'])
        self.summary = self.trainer_parameters['summary_path']
        self.tensorBoard = tf.summary.FileWriter(logdir=self.summary)
        self.model = None

    def __str__(self):
        return '''DQN Trainer'''

    @property
    def parameters(self):
        """
        Returns the trainer parameters of the trainer.
        """
        return self.trainer_parameters

    @property
    def get_max_steps(self):
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        return self.trainer_parameters['max_steps']

    @property
    def get_step(self):
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self.steps

    @property
    def get_last_reward(self):
        """
        Returns the last reward the trainer has had
        :return: the new last reward
        """
        return self.last_reward

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = np.zeros((self.state_size, self.action_size))
        return model

    def is_initialized(self):
        """
        check if the trainer is initialized
        """
        return self.initialized

    def initialize(self):
        """
        Initialize the trainer
        """
        self.model = self._build_model()
        print('Summary SARSA Q-Matrix ({},{}): \n{}\n'.format(self.state_size, self.action_size, self.model))
        self.initialized = True

    def clear(self):
        """
        Clear the trainer
        """
        k.clear_session()
        self.replay_memory.clear()
        self.model = None

    def load_model_and_restore(self, model_path):
        """
        Load and restore the model from a defined path.

        :param model_path: saved model.
        """
        if os.path.exists('./' +model_path + '/SARSA.h5.npy'):
            self.model = self._build_model()
            self.model = np.load('./' +model_path + '/SARSA.h5.npy')
            print('Summary SARSA Q-Matrix ({},{}): \n{}\n'.format(self.state_size, self.action_size, self.model))
        else:
            self.model = self._build_model()

    def increment_step(self):
        """
        Increment the step count of the trainer
        """
        self.steps = self.steps + 1

    def update_last_reward(self, rewards):
        """
        Updates the last reward
        """
        self.last_reward = rewards

    def take_action(self, observation, _env):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param observation: The BrainInfo from environment.
        :param _env: The environment.
        :return: the action array and an object as cookie
        """
        if self.is_training and np.random.rand() <= self.epsilon:
            return np.argmax(np.random.randint(0, 2, self.action_size))
        else:
            _max = np.nanmax(self.model[np.argmax(observation)])
            indices = np.argwhere(self.model[np.argmax(observation)] == _max)
            choice = np.random.choice(indices.size)
            return indices[choice, 0]

    def add_experiences(self, observation, action, next_observation, reward, done, info):
        """
        Adds experiences to each agent's experience history.
        :param observation: the observation before executing the action
        :param action: Current executed action
        :param next_observation: the observation after executing the action
        :param reward: the reward obtained after executing the action.
        :param done: true if the episode ended.
        :param info: info after executing the action.
        """
        self.replay_memory.append(
            (observation, action, next_observation, reward, done, info))

    def process_experiences(self, current_info, action_vector, next_info):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Current BrainInfo.
        :param action_vector: Current executed action
        :param next_info: Next corresponding BrainInfo.
        """
        # Nothing to be done in the DQN case
        pass

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        # self.replay_memory.clear()
        pass

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        # The NN is ready to be updated if there is at least a batch in the replay memory
        # return (len(self.replay_memory) >= self.batch_size) and (len(self.replay_memory) % self.batch_size == 0)

        # The NN is ready to be updated everytime a batch is sampled
        return (self.steps > 1) and ((self.steps % self.batch_size) == 0)

    def update_model(self):
        """
        Uses the memory to update model. Run back propagation.
        """
        num_samples = min(self.batch_size, len(self.replay_memory))
        mini_batch = random.sample(self.replay_memory, num_samples)

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        for state, action, next_state, reward, done, info in mini_batch:
            next_action = np.argmax(self.model[np.argmax(next_state)])
            delta = self.alpha * \
                           (
                                   reward +
                                   (self.gamma * self.model[np.argmax(next_state), next_action])
                                   - self.model[np.argmax(state), action]
                           )
            self.model[np.argmax(state), action] = self.model[np.argmax(state), action] + delta
        # TODO: check the performance with the following trick - Jupiter
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.alpha > self.alpha_min:
            self.alpha *= self.alpha_decay

        print('Actual Q-Matrix ({},{}): \n{}\n'.format(self.state_size, self.action_size, self.model))

    def save_model(self, model_path):
        """
        Save the model architecture.
        :param model_path: The path where the model will be saved.
        """
        if os.path.exists(model_path):
            np.save(model_path + '/SARSA.h5', self.model)
        else:
            raise FAPSTrainerException("The model path doesn't exist. model_path : " + model_path)

    def write_summary(self):
        """
        Saves training statistics to i.e. Tensorboard.
        """
        # TODO: Add Tensorboard support - Jupiter
        pass

    def write_tensor_board_text(self, key, input_dict):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        """
        try:
            s_op = tf.summary.text(key,
                                   ([[str(x), str(input_dict[x])] for x in input_dict])
                                   )

            self.tensorBoard.add_summary(s_op, int(self.steps / self.batch_size))
        except:
            logger.info("Cannot write text summary for Tensorboard. Tensorflow version must be r1.2 or above.")

    def write_tensorboard_value(self, key, value):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param value: A value that will bw displayed on Tensorboard.
        """
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = key
        summary_value.node_name = key
        self.tensorBoard.add_summary(summary, int(self.steps / self.batch_size))
        self.tensorBoard.flush()

    @staticmethod
    def _write_log(callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            summary_value.node_name = name
            callback.add_summary(summary, batch_no)
            callback.flush()


pass
