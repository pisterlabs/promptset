# # FAPS PLMAgents

import logging
import os
import random
from collections import deque

import keras.backend as k
import numpy as np
import tensorflow as tf
from keras import Input
from keras.layers import Dense, add, Dropout
from keras.models import Model
from keras.optimizers import Adam

from OpenAIGym.FAPSPLMAgents.utils.random import OrnsteinUhlenbeckProcess
from OpenAIGym.exception import FAPSPLMEnvironmentException

logger = logging.getLogger("FAPSPLMAgents")


class FAPSTrainerException(FAPSPLMEnvironmentException):
    """
    Related to errors with the Trainer.
    """
    pass


class MADDPG(object):
    """This class is the abstract class for the faps trainers"""

    def __init__(self, envs, brain_name, trainer_parameters, training, seed):
        """
        Responsible for collecting experiences and training a neural network model.

        :param envs: The FAPSPLMEnvironment.
        :param brain_name: The brain to train.
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param seed: Random seed.
        """
        self.brain_name = brain_name
        self.trainer_parameters = trainer_parameters
        self.is_training = training
        self.seed = seed
        self.steps = 0
        self.last_reward = 0
        self.initialized = False

        # initialize specific MADDPG parameters
        self.time_slice = 4
        self.env_brains = envs
        self.state_size = 0
        self.action_size = 0

        # number of agents
        self.agent_count = 0

        # Initialize the environment
        self.agent_config_path = self.trainer_parameters['agent_config_path']
        for env_name, env in self.env_brains.items():
            env.configure(display=True, agent_config_path=self.agent_config_path,
                          shared_reward=False)
            self.agent_count = env.n

        self.action_size = [None] * self.agent_count
        self.state_size = [None] * self.agent_count

        for env_name, env in self.env_brains.items():
            for i in range(self.agent_count):
                self.action_size = env.action_space[i].n
                self.state_size = env.observation_space[i].n

        # self.action_space_type = envs.actionSpaceType
        self.num_layers = self.trainer_parameters['num_layers']
        self.batch_size = self.trainer_parameters['batch_size']
        self.hidden_units = self.trainer_parameters['hidden_units']
        self.replay_memory = deque(maxlen=self.trainer_parameters['memory_size'])

        self.replay_sequence = []
        for i in range(self.agent_count):
            self.replay_sequence.append(deque(maxlen=self.time_slice))

        self.gamma = self.trainer_parameters['gamma']  # discount rate
        self.epsilon = self.trainer_parameters['epsilon']  # exploration rate
        self.epsilon_min = self.trainer_parameters['epsilon_min']
        self.epsilon_decay = self.trainer_parameters['epsilon_decay']
        self.alpha = self.trainer_parameters['alpha']
        self.alpha_decay = self.trainer_parameters['alpha_decay']
        self.alpha_min = self.trainer_parameters['alpha_min']
        self.learning_rate = self.trainer_parameters['learning_rate']
        self.tau = self.trainer_parameters['tau']
        self.summary = self.trainer_parameters['summary_path']
        self.tensorBoard = tf.summary.FileWriter(logdir=self.summary)
        self.actor_model = [None] * self.agent_count
        self.target_model = [None] * self.agent_count
        self.critic_model = None
        self.critic_target_model = None
        self.critic_gradient_wrt_action = None  # GRADIENTS for policy update
        self.critic_gradient_wrt_action_fn = None
        self.actor_apply_gradient_fn = [None] * self.agent_count
        self.critic_state = None
        self.critic_action = None
        self.random_process = []
        self.actor_trainable_weights = [None] * self.agent_count
        self.actor_input = [None] * self.agent_count
        self.action_gradient = [None] * self.agent_count

    def __str__(self):
        return '''Multi Agent Deep Deterministic Policy Gradient Trainer'''

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
        return self.trainer_parameters['max_steps'] * self.trainer_parameters['num_epoch']

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

    def get_actor_optimizer(self, index):
        """
        Actor Optimizer
        """
        action_gdts = k.placeholder(shape=(None, self.action_size[index]))
        params_grad = tf.gradients(self.actor_model[index].output, self.actor_model[index].trainable_weights,
                                   -action_gdts)
        grads = zip(params_grad, self.actor_model[index].trainable_weights)
        return k.function([self.actor_model[index].input, action_gdts],
                          [tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)][1:])

    def _create_actor_model(self, index):
        a = Input(shape=[self.state_size[index] * self.time_slice])
        h = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(a)
        h = Dropout(0.2)(h)
        for x in range(1, self.num_layers):
            h = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(h)
            h = Dropout(0.2)(h)
        o = Dense(self.action_size[index], activation='softmax', kernel_initializer='he_uniform')(h)
        model = Model(inputs=a, outputs=o)
        return model, model.trainable_weights, a

    def _create_critic_model(self):
        ## TODO: Update
        s = Input(shape=[self.state_size * self.time_slice], name='critic_state')
        a = Input(shape=[self.action_size], name='critic_action')
        h1 = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(s)
        h1 = Dropout(0.2)(h1)
        # h1 = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(h1)
        # h1 = Dropout(0.2)(h1)
        a1 = Dense(self.hidden_units, activation='relu')(a)
        a1 = Dropout(0.2)(a1)

        h2 = add([h1, a1])
        for x in range(1, self.num_layers):
            h2 = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(h2)
            h2 = Dropout(0.2)(h2)

        v = Dense(1, activation='tanh', kernel_initializer='he_uniform')(h2)
        model = Model(inputs=[s, a], outputs=v)

        return model, s, a

    def is_initialized(self):
        """
        check if the trainer is initialized
        """
        return self.initialized

    def initialize(self):
        """
        Initialize the trainer
        """
        # Initialize the actors of all agents
        for i in range(self.agent_count):
            self.actor_model[i], self.actor_trainable_weights[i], self.actor_input[i] = self._create_actor_model(i)
            self.actor_model[i].compile(loss='mse', metrics=['mse'], optimizer=Adam(lr=self.learning_rate))
            print('\n##### Actors Model {}'.format(i))
            print(self.actor_model[i].summary())

            self.target_model[i], target_actor_trainable_weights, target_actor_input = self._create_actor_model(i)
            self.target_model[i].compile(loss='mse', metrics=['mse'], optimizer=Adam(lr=self.learning_rate))
            print('\n##### Actor Target Model {}'.format(i))
            print(self.target_model[i].summary())

        # initialize the global critic
        # TODO: Critic
        self.critic_target_model, target_critic_state_input, target_critic_action_input = self._create_critic_model()
        self.critic_target_model.compile(loss='mse', metrics=['mse'], optimizer=Adam(lr=self.learning_rate))
        print('\n##### Critic Target Model ')
        print(self.critic_target_model.summary())

        self.critic_model, self.critic_state, self.critic_action = self._create_critic_model()
        self.critic_model.compile(loss='mse', metrics=['mse'], optimizer=Adam(lr=self.learning_rate))
        print('\n##### Critic Model ')
        print(self.critic_model.summary())

        # Create a function that calculate the critic gradient for policy update
        self.critic_gradient_wrt_action = k.gradients(self.critic_model.output, self.critic_action)
        if k.backend() == 'tensorflow':
            self.critic_gradient_wrt_action_fn = k.function([self.critic_state, self.critic_action],
                                                            self.critic_gradient_wrt_action)
        else:
            raise RuntimeError('Unknown backend "{}".'.format(k.backend()))

        # Create a function that update the gradient of the actor
        if k.backend() == 'tensorflow':
            for i in range(self.agent_count):
                self.actor_apply_gradient_fn[i] = self.get_actor_optimizer(i)
        else:
            raise RuntimeError('Unknown backend "{}".'.format(k.backend()))

        # Define a random process
        for j in range(self.agent_count):
            self.random_process.append(OrnsteinUhlenbeckProcess(size=self.action_size, theta=.10, mu=0., sigma=0.5))

        self.initialized = True

    def clear(self):
        """
        Clear the trainer
        """
        k.clear_session()
        self.replay_memory.clear()
        self.actor_model = [None] * self.agent_count
        self.target_model = [None] * self.agent_count
        self.critic_model = None
        self.critic_target_model = None

    def load_model_and_restore(self, model_path):
        """
        Load and restore the model from a defined path.

        :param model_path: Random seed.
        """
        for i in range(self.agent_count):
            self.actor_model[i], self.actor_trainable_weights[i], self.actor_input[i] = self._create_actor_model(i)
            if os.path.exists('./' + model_path + '/MADDPG_actor_{}_model.h5'.format(i)):
                self.actor_model[i].load_weights('./' + model_path + '/MADDPG_actor_{}_model.h5'.format(i))
            self.actor_model[i].compile(loss='mse', metrics=['mse'], optimizer=Adam(lr=self.learning_rate))

            self.target_model[i], t_weigth, t_input = self._create_actor_model(i)
            if os.path.exists('./' + model_path + '/MADDPG_actor_{}_target_model.h5'.format(i)):
                self.target_model[i].load_weights('./' + model_path + '/MADDPG_actor_{}_target_model.h5'.format(i))
            self.target_model[i].compile(loss='mse', metrics=['mse'], optimizer=Adam(lr=self.learning_rate))

        self.critic_target_model, target_critic_state_input, target_critic_action_input = self._create_critic_model()
        if os.path.exists('./' + model_path + '/MADDPG_critic_target_model.h5'):
            self.critic_target_model.load_weights('./' + model_path + '/MADDPG_critic_target_model.h5')
        self.critic_target_model.compile(loss='mse', metrics=['mse'], optimizer=Adam(lr=self.learning_rate))

        self.critic_model, self.critic_state, self.critic_action = self._create_critic_model()
        if os.path.exists('./' + model_path + '/MADDPG_critic_model.h5'):
            self.critic_model.load_weights('./' + model_path + '/MADDPG_critic_model.h5')
        self.critic_model.compile(loss='mse', metrics=['mse'], optimizer=Adam(lr=self.learning_rate))

    def increment_step(self):
        """
        Increment the step count of the trainer
        """
        self.steps = self.steps + 1

    def update_last_reward(self, reward):
        """
        Updates the last reward
        """
        self.last_reward = reward[0]

    def take_action(self, observation, _env):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param observation: The BrainInfo from environment.
        :param _env: The environment.
        :return: the action array and an object as cookie
        """
        _actions = []
        for i in range(self.agent_count):
            if len(self.replay_sequence[i]) < self.time_slice - 1:
                _actions.append(np.argmax(np.random.randint(0, 2, self.action_size[i])))
            else:
                last_elements = self.replay_sequence[i].copy()
                last_elements.append(observation[i][0])
                arr_last_elements = np.array(last_elements)
                tmp = arr_last_elements.reshape((1, self.state_size[i] * self.time_slice))
                act_values = self.actor_model[i].predict(tmp)
                if self.is_training:
                    noise = self.random_process[i].sample()
                    act_values += noise

                # Make discrete action
                _max = np.nanmax(act_values[0])
                indices = np.argwhere(act_values[0] == _max)
                choice = np.random.choice(indices.size)

                _actions.append(indices[choice, 0])

        return _actions

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
        self.replay_sequence.append(observation[0])
        if len(self.replay_sequence) >= self.time_slice:
            tmp = np.array(self.replay_sequence.copy()).reshape((1, self.state_size * self.time_slice))

            next_last_elements = self.replay_sequence.copy()
            next_last_elements.append(next_observation[0])
            next_arr_last_elements = np.array(next_last_elements)
            next_tmp = next_arr_last_elements.reshape((1, self.state_size * self.time_slice))

            self.replay_memory.append((tmp, action[0], next_tmp, reward[0], done[0], info[0]))

    def process_experiences(self, current_info, action_vector, next_info):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Current BrainInfo.
        :param action_vector: Current executed action
        :param next_info: Next corresponding BrainInfo.
        """
        # Nothing to be done in the DQN case

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        # print("End Episode...")

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        # The NN is ready to be updated if there is at least a batch in the replay memory
        # return (len(self.replay_memory) >= self.batch_size) and (len(self.replay_memory) % self.batch_size == 0)

        # The NN is ready to be updated everytime a batch is sampled
        return (self.steps > 1) and ((self.steps % self.batch_size) == 0)

    def _copy_target_models(self):
        critic_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

        actor_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def update_model(self):
        """
        Uses the memory to update model. Run back propagation.
        """
        # TODO: update to support multiple agents. Now only one agent is supported
        num_samples = min(self.batch_size, len(self.replay_memory))
        mini_batch = random.sample(self.replay_memory, num_samples)

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for state, action, next_state, reward, done, info in mini_batch:
            state0_batch.append(state)
            state1_batch.append(next_state)
            reward_batch.append(reward)
            action_batch.append(action)
            terminal1_batch.append(0. if done else 1.)

        state0_batch = np.array(state0_batch).reshape((num_samples, self.state_size * self.time_slice))
        state1_batch = np.array(state1_batch).reshape((num_samples, self.state_size * self.time_slice))
        # terminal1_batch = np.array(terminal1_batch).reshape((num_samples, 1))
        reward_batch = np.array(reward_batch).reshape((num_samples, 1))
        action_batch = np.array(action_batch).reshape((num_samples, 1))

        # Update critic
        target_q_values = self.critic_target_model.predict_on_batch([state1_batch,
                                                                     self.actor_model.predict_on_batch(state1_batch)])
        self.write_tensorboard_value('target_v_values', target_q_values.mean())
        discounted_reward_batch = self.gamma * target_q_values
        # discounted_reward_batch = discounted_reward_batch * terminal1_batch
        y_critic_t = (reward_batch + discounted_reward_batch).reshape(num_samples, 1)
        train_action_batch = np.zeros((num_samples, self.action_size))
        np.put_along_axis(arr=train_action_batch, indices=action_batch, values=1, axis=1)
        logs = self.critic_model.train_on_batch([state0_batch, train_action_batch], y_critic_t)
        self.write_tensorboard_value('y_critic_t', y_critic_t.mean())

        train_names = ['critic_train_loss', 'critic_train_mse']
        self._write_log(self.tensorBoard, train_names, logs, int(self.steps / self.batch_size))

        # Update actor
        # Compute the gradient from critic
        a_for_grad = self.actor_model.predict(state0_batch)
        critic_gradients = self.critic_gradient_wrt_action_fn([state0_batch, a_for_grad])[0]
        assert critic_gradients.shape == (num_samples, self.action_size)

        # apply the critic gradient to the actor model
        self.actor_apply_gradient_fn([state0_batch, critic_gradients])

        # Copy target model
        self._copy_target_models()

        self.write_tensorboard_value('cul_reward_mean', reward_batch.mean())
        ############################################################################

    def save_model(self, model_path):
        """
        Save the model architecture to i.e. Tensorboard.
        :param model_path: The path where the model will be saved.
        """
        if os.path.exists('./' + model_path):
            for i in range(self.agent_count):
                self.actor_model[i].save('./' + model_path + '/MADDPG_actor_{}_model.h5'.format(i))
                self.target_model[i].save('./' + model_path + '/MADDPG_actor_{}_target_model.h5'.format(i))

            self.critic_model.save('./' + model_path + '/MADDPG_critic_model.h5')
            self.critic_target_model.save('./' + model_path + '/MADDPG_critic_target_model.h5')
        else:
            raise FAPSTrainerException("The model path doesn't exist. model_path : " + './' + model_path)

    def write_summary(self):
        """
        Saves training statistics to i.e. Tensorboard.
        """
        # print(self.model.summary())
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
        self.tensorBoard.add_summary(summary, int(self.steps / self.batch_size))
        self.tensorBoard.flush()

    @staticmethod
    def _write_log(callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.add_summary(summary, batch_no)
            callback.flush()

pass
