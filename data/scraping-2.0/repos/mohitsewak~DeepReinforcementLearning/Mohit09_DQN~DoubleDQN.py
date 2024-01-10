""" DQN in Code - BehaviorPolicy

DQN Code as in the book Deep Reinforcement Learning, Chapter 9.

Runtime: Python 3.6.5
Dependencies: numpy, matplotlib, tensorflow (/ tensorflow-gpu), keras
DocStrings: GoogleStyle

Author : Mohit Sewak (p20150023@goa-bits-pilani.ac.in)

"""

# make the general imports. Many of these libraries come bundled in miniconda/ base-python and hence are excluded from
# requirements.txt
import logging
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import time
import os
# Make the imports from Keras for making Deep Learning model. Keras is a wrapper to some of the popular deeplearning
# libraries like tensorflow, theano, mnist. One of these needs to be installed for keras to work. We are using
# tensorflow as is indicated in requirements.txt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import mean_squared_error
# We will require an environment for the agent to work. This can be provided from an external code take instantiates
# the agent, and is not required here in that case. We can use both a custom environment or one from OpenAI gym. A
# custom environment may require some changes in the n_states, and n_action parameters to be compatible
import gym
# Last we import other cutom dependencies that we have coded in external modules to make this code small, simple
# to understand, easy to maintain, and modular. For example we use the epsilon_decay policy instead of epsilon_greedy
# in this code. So the policy and memory are separate modules which can be enhanced, new ones implemented, and used
# as requirement as standard implementation for multiple agents.
from experience_replay import SequentialDequeMemory
from beahaviour_policy import BehaviorPolicy

# Configure logging for the project
# Create file logger, to be used for deployment
# logging.basicConfig(filename="Chapter09_DDQN.log", format='%(asctime)s %(message)s', filemode='w')
logging.basicConfig()
# Creating a stream logger for receiving inline logs
logger = logging.getLogger()
# Setting the logging threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


class DoubleDQN:
    """Double DQN Agent

        Class for training a Double DQN Learning agent on any custom environment.

        Args:
            agent_name (str): The name of the agent. This argument helps in continuing the training from the last auto
                            checkpoint. The system will check if any agent's weight by the same name exists, if so then
                            the existing weights are saved before resuming/ starting training.
            env (Object): An object instantiation of a OpenAI gym compatible env class like the CartPole-v1 environment
            number_episodes (int): The maximum number of episodes to be executed for training the agent
            discounting_factor (float): The discounting factor (gamma) used to discount the future rewards to current step
            learning_rate (float): The learning rate (alpha) used to update the q values in each step
            behavior_policy (str): The behavior policy chosen (as q learning is off policy). Example "epsilon-greedy"
            policy_parameters (dict): A dict with the behavior policy parameters. The keys required for epsilon_greedy is
                            just epsilon, and for epsilon_decay additionally requires min_epsilon and epsilon_decay_rate
            deep_learning_model_hidden_layer_configuration (list): A list if integers corresponding to the number of
                        neurons in each hidden layer of the MLP-DNN network for the model.

        Examples:
            agent = DoubleDQN()

    """

    def __init__(self, agent_name=None, env=gym.make('CartPole-v1'), number_episodes = 500, discounting_factor = 0.9,
                 learning_rate = 0.001, behavior_policy = "epsilon_decay",
                 policy_parameters={"epsilon":1.0,"min_epsilon":0.01,"epsilon_decay_rate":0.99},
                 deep_learning_model_hidden_layer_configuration=[32,16,8]):
        self.agent_name = "ddqa_"+str(time.strftime("%Y%m%d-%H%M%S")) if agent_name is None else agent_name
        self.model_weights_dir = "model_weights"
        self.env = env
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.n_episodes = number_episodes
        self.episodes_completed = 0
        self.gamma = discounting_factor
        self.alpha = learning_rate
        self.policy = BehaviorPolicy(n_actions=self.n_actions, policy_type=behavior_policy,
                                     policy_parameters=policy_parameters).getPolicy()
        self.policyParameter = policy_parameters
        self.model_hidden_layer_configuration = deep_learning_model_hidden_layer_configuration
        self.online_model = self._build_sequential_dnn_model()
        self.target_model = self._build_sequential_dnn_model()
        self.trainingStats_steps_in_each_episode = []
        self.trainingStats_rewards_in_each_episode = []
        self.trainingStats_discountedrewards_in_each_episode = []
        self.memory = SequentialDequeMemory(queue_capacity=2000)
        self.experience_replay_batch_size = 32

    def _build_sequential_dnn_model(self):
        """Internal helper function for building DNN model

            This function can make a custom MLP-DNN topology based on the arguments provided during instantiation of
            the class.
            The MLP-DNN model starts with the input layer, with as many nodes as the state cardinality,
            then add as many hidden layers with as many neurons in each hidden layer as requested in the instantiation
            parameter self.model_hidden_layer_configuration. Then th output layer with as many layers as
            action space cardinality follows.
            The activation for input and hidden layers is ReLU, and for output payer neurons is Linear.
            Optimizer used is ADAM, and Loss function used is Mean Square Error (MSE)

        """

        model = Sequential()
        hidden_layers = self.model_hidden_layer_configuration
        model.add(Dense(hidden_layers[0], input_dim=self.n_states, activation='relu'))
        for layer_size in hidden_layers[1:]:
            model.add(Dense(layer_size, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=self.alpha))
        return model

    def _sync_target_model_with_online_model(self):
        """Internal helper function to sync the target Q network with the online Q network
        """
        self.target_model.set_weights(self.online_model.get_weights())

    def _update_online_model(self,experience_tuple):
        """Internal helper function for updating the online Q network
        """
        current_state, action, instantaneous_reward, next_state, done_flag = experience_tuple
        action_target_values = self.online_model.predict(current_state)
        action_values_for_state = action_target_values[0]
        if done_flag:
            action_values_for_state[action] = instantaneous_reward
        else:
            action_values_for_next_state = self.target_model.predict(next_state)[0]
            max_next_state_value = np.max(action_values_for_next_state)
            target_action_value = instantaneous_reward + self.gamma * max_next_state_value
            action_values_for_state[action] = target_action_value
        action_target_values[0] = action_values_for_state
        logger.debug("Fitting online model with Current_State: {}, Action_Values: {}".
                     format(current_state,action_target_values))
        self.online_model.fit(current_state, action_target_values, epochs=1)

    def _reshape_state_for_model(self,state):
        """Internal helper function for shaping state to be compatible with the DNN model
        """
        return np.reshape(state,[1,self.n_states])

    def train_agent(self):
        """Main function to train the agent

             The main function that needs to be called to start the training of the agent after instantiating it.

            Returns:
                tuple: Tuple of 3 lists, 1st is the steps in each episode, 2nd is the total un-discounted rewards in
                        each episode, and 3rd is the total discounted rewards in each episode.

            Examples:
                training_statistics = agent.train_agent()

        """

        self.load_model_weights()
        for episode in range(self.n_episodes):
            logger.debug("-"*30)
            logger.debug("EPISODE {}/{}".format(episode,self.n_episodes))
            logger.debug("-"*30)
            current_state = self._reshape_state_for_model(self.env.reset())
            cumulative_reward = 0
            discounted_cumulative_reward = 0
            for n_step in count():
                all_action_value_for_current_state = self.online_model.predict(current_state)[0]
                policy_defined_action = self.policy(all_action_value_for_current_state)
                next_state, instantaneous_reward, done, _ = self.env.step(policy_defined_action)
                next_state = self._reshape_state_for_model(next_state)
                experience_tuple = (current_state, policy_defined_action, instantaneous_reward, next_state, done)
                self.memory.add_to_memory(experience_tuple)
                cumulative_reward += instantaneous_reward
                discounted_cumulative_reward = instantaneous_reward + self.gamma * discounted_cumulative_reward
                if done:
                    self.trainingStats_steps_in_each_episode.append(n_step)
                    self.trainingStats_rewards_in_each_episode.append(cumulative_reward)
                    self.trainingStats_discountedrewards_in_each_episode.append(discounted_cumulative_reward)
                    self._sync_target_model_with_online_model()
                    logger.debug("episode: {}/{}, reward: {}, discounted_reward: {}".format(n_step, self.n_episodes,
                                                                    cumulative_reward, discounted_cumulative_reward))
                    break
                self.replay_experience_from_memory()
        if episode % 2 == 0: self.plot_training_statistics()
        if episode % 5 == 0: self.save_model_weights()
        return self.trainingStats_steps_in_each_episode, self.trainingStats_rewards_in_each_episode, \
               self.trainingStats_discountedrewards_in_each_episode

    def replay_experience_from_memory(self):
        """Replays the experience from memory buffer

             Replays the experience from the memory buffer. The memory buffer is as selected during the class
             instantiation.

            Returns:
                bool: True if the replay happens, False if the size of buffer is less than the batch size and hence
                        replay does not happens.

        """

        if self.memory.get_memory_size() < self.experience_replay_batch_size:
            return False
        experience_mini_batch = self.memory.get_random_batch_for_replay(batch_size=self.experience_replay_batch_size)
        for experience_tuple in experience_mini_batch:
            self._update_online_model(experience_tuple)
        return True

    def save_model_weights(self, agent_name=None):
        """Save Model Weights

            Saves the model weights for both the target and online Q network model from the directory given in class
            variable self.model_weights_dir (default = model_weights) and adds .h5 extension.

            Args:
                agent_name (str): Name of the agent if need to be forced a specific one other than the default unique one

            Returns:
                None

        """

        if agent_name is None:
            agent_name = self.agent_name
        model_file = os.path.join(os.path.join(self.model_weights_dir,agent_name+".h5"))
        self.online_model.save_weights(model_file, overwrite=True)

    def load_model_weights(self, agent_name=None):
        """Load Model Weights

            Loads the model weights for both the target and online Q network model from the directory given in class
            variable self.model_weights_dir (default = model_weights) and the one has .h5 extension.

            Args:
                agent_name (str): Name of the agent if need to be forced a specific one other than the default unique one

            Returns:
                None

        """

        if agent_name is None:
            agent_name = self.agent_name
        model_file = os.path.join(os.path.join(self.model_weights_dir, agent_name + ".h5"))
        if os.path.exists(model_file):
            self.online_model.load_weights(model_file)
            self.target_model.load_weights(model_file)

    def plot_training_statistics(self, training_statistics = None):
        """Plot Training Statistics

            Function to plot training statistics of the Q Learning agent's training. This function plots the dual axis
            plot, with the episode count on the x axis and the steps and rewards in each episode on the y axis.

            Args:
                training_statistics (tuple): Tuple of list of steps, list or rewards, list of cumulative rewards for
                                            each episode

            Returns:
                None

            Examples:
                agent.plot_statistics()

        """

        steps = self.trainingStats_steps_in_each_episode if training_statistics is None else training_statistics[0]
        rewards = self.trainingStats_rewards_in_each_episode if training_statistics is None else training_statistics[1]
        discounted_rewards = self.trainingStats_discountedrewards_in_each_episode if training_statistics is None \
            else training_statistics[2]
        episodes = np.arange(len(self.trainingStats_steps_in_each_episode))
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Episodes (e)')
        ax1.set_ylabel('Steps To Episode Completion', color="red")
        ax1.plot(episodes, steps, color="red")
        ax2 = ax1.twinx()
        ax2.set_ylabel('Reward in each Episode', color="blue")
        ax2.plot(episodes, rewards, color="blue")
        fig.tight_layout()
        plt.show()
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Episodes (e)')
        ax1.set_ylabel('Steps To Episode Completion', color="red")
        ax1.plot(episodes, steps, color="red")
        ax2 = ax1.twinx()
        ax2.set_ylabel('Discounted Reward in each Episode', color="green")
        ax2.plot(episodes, discounted_rewards, color="green")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    """Main function

        A sample implementation of the above Double DQN agent for testing purpose.
        This function is executed when this file is run from the command prompt directly or by selection.

    """

    agent = DoubleDQN()
    training_statistics = agent.train_agent()
    agent.plot_training_statistics(training_statistics)
