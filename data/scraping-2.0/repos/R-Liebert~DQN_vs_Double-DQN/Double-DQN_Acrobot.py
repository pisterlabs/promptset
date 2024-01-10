import random
import datetime
import numpy as np
from gym import wrappers

from collections import deque
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import gym
from gym import wrappers

import os

'''
Original paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- DQN model with Dense layers only
- Model input is changed to take current and n previous states where n = time_steps
- Multiple states are concatenated before given to the model
- Uses target model for more stable training
- More states was shown to have better performance for CartPole env

This is the Double-DQN script from OpenAI Baseline which has been modified to work with the updated CartPole-v0 environment.
And edited to be a vanilla DQN. Sigopt is used to tune the hyperparameters.

'''


class MyModel(tf.keras.Model):
    """
    In the call method, we define the forward pass of the model.
    The model can be given a width and depth, inputs shape , output shape and learning rate.

    Arguments:
    ----------
    num_states: int
        The number of states in the environment
    hidden_units: int
        The number of hidden units in the model
    hidden_layers: int
        The number of hidden layers in the model
    num_actions: int
        The number of actions in the environment
    lr: float
        The learning rate of the model

    Returns:
    --------
    none
    """
    def __init__(self, num_states, hidden_units, hidden_layers, num_actions, lr):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for _ in range(hidden_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(
                hidden_units, activation='relu', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')
        
    @tf.function
    def call(self, inputs):
        """
        Simple forward pass of the model

        Arguments:
        ----------
        inputs: array
            The input to the model
        
        Returns:
        --------
        output: array
            The output of the model
        """
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN:
    """
    DQN agent with target model and experience replay
    The DQN agent that interacts with the environment. It has a memory buffer that stores the past experiences.
    It also has a the neural network that is used to predict the Q values of the states.
    
    Arguments:
    ----------
    num_states: int
        The number of states in the environment
    num_actions: int
        The number of actions in the environment
    hidden_units: int
        The number of hidden units in the model
    hidden_layers: int
        The number of hidden layers in the model
    gamma: float
        The discount factor
    max_experiences: int
        The maximum number of experiences to store in the memory buffer
    min_experiences: int
        The minimum number of experiences to store in the memory buffer before training
    batch_size: int
        The number of experiences to sample from the memory buffer for training
    lr: float
        The learning rate of the model
    max_steps: int
        The maximum number of steps to run the environment for
    decay_rate: float
        The decay rate of the learning rate
    
    
    Returns:
    --------
    none
    """
    def __init__(self, num_states, num_actions, hidden_units, hidden_layers, gamma, max_experiences, min_experiences, batch_size, lr, max_steps, decay_rate):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(learning_rate=lr, decay=decay_rate)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, hidden_layers, num_actions, lr)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.max_steps = max_steps

    def predict(self, inputs):
        """
        Predicts the Q values of the states
        
        Arguments:
        ----------
        inputs: array
            The states of the environment
            
        Returns:
        --------
        output: array
            The Q values of the states
        """
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet):
        """
        Trains the model using the experiences in the memory buffer
        
        Arguments:
        ----------
        TrainNet: keras model
            The target model
            
        Returns:
        --------
        loss: float
            The loss of the model
        """
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        """
        Gets the action to take based on the epsilon greedy policy
        
        Arguments:
        ----------
        states: array
            The states of the environment
        epsilon: float
            The probability of taking a random action
            
        Returns:
        --------
        action: int
            The action to take
        """
        valid_actions = [a for a in range(self.num_actions)] 
        if np.random.random() < epsilon:
            return np.random.choice(valid_actions)
        else:
            best_action = np.argmax(self.predict(np.atleast_2d(states))[0])
            if best_action in valid_actions:
                return best_action
            else:
                return np.random.choice(valid_actions)

    def add_experience(self, exp):
        """
        Adds the experience to the memory buffer
        
        Arguments:
        ----------
        exp: tuple
            The experience to add to the memory buffer
            
        Returns:
        --------
        none
        """
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        """
        Copies the weights of the model to the target model
        
        Arguments:
        ----------
        TrainNet: keras model
            The target model
            
        Returns:
        --------
        none
        """
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    """
    Here the interaction between the agent and the environment takes place.
    The agent takes an action, the environment returns the next state, reward and done.
    The agent then adds the experience to the memory buffer and trains the model.
    The target model is updated every copy_step.

    Arguments:
    ----------
    env: gym environment
        The environment to interact with
    TrainNet: keras model
        The target model
    epsilon: float
        The probability of taking a random action

    Returns:
    --------
    total_reward: float
        The total reward obtained from the episode
    """
    rewards = 0
    iter = 0
    done = False
    truncated = False
    observations, _ = env.reset()
    losses = list()
    while not done and not truncated:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, truncated, _ = env.step(action)
        rewards += reward
        if done or truncated:
            env.reset()
            if truncated:
                rewards = -500

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)

    return rewards, np.mean(losses)

def test(env, TrainNet):
    """
    Tests the model on the environment

    Arguments:
    ----------
    env: gym environment
        The environment to interact with
    TrainNet: keras model
        The target model
    
    Returns:
    --------
    rewards: int
        The total reward obtained from the episode
    mean_loss: float
        The mean loss of the model
    """
    rewards = 0
    steps = 0
    done = False
    truncated = False
    observations, _ = env.reset()
    while not done and not truncated:
        action = TrainNet.get_action(observations, 0)
        observations, reward, done, truncated, _= env.step(action)
        steps += 1
        rewards += reward
    
    rewards = -500 if truncated==True else rewards 

    return rewards


def main():
    """
    Here we initialize the environment, agent and train the agent.
    The first part is checking if there is a GPU available.
    If you have GPU's, you're a lucky bitch, and can uncomment the GPU line.

    First all hyperparameters that should be tuned for optimum performance are set.
    Then the loop goes until max number of episodes are reached or the environment is solved
    Last some metrics are printed
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    os.environ["CUDA_VISIBLE_DEVICES"]="0"  # use GPU with ID=0 (uncomment if GPU is available)


    env = gym.make('Acrobot-v1')


    max_steps = 500 # Environment max step
    env._max_episode_steps = max_steps

    gamma = 0.95 
    copy_step = 25
    num_states = 6 # hardcoded until solved
    num_actions = 3 # hardcoded until solved
    hidden_units = 2
    hidden_layers = 24
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-3
    decay_rate = 0.95
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    TrainNet = DQN(num_states, num_actions, hidden_units, hidden_layers, gamma, max_experiences, min_experiences, batch_size, lr, max_steps, decay_rate)
    TargetNet = DQN(num_states, num_actions, hidden_units, hidden_layers, gamma, max_experiences, min_experiences, batch_size, lr, max_steps, decay_rate)
    max_episodes = 1000
    total_rewards = np.empty(max_episodes)
    epsilon = 1
    decay = 0.99
    min_epsilon = 0.1
    for episode in range(max_episodes):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[episode]  = total_reward
        avg_rewards = np.mean(total_rewards[max(0, episode - 100):(episode + 1)])
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=episode)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=episode)
            tf.summary.scalar('average loss)', losses, step=episode)
        if episode % 100 == 0 and episode != 0:
            print(f"episode: {episode}, episode reward: {total_reward}, eps: {epsilon}, avg reward (last 100): {avg_rewards}, episode loss: {losses}")
        # Check if last 100 episodes have total_reward >= 195 to approve training
        if episode >= 100 and all(total_rewards[max(0, episode - 100):(episode + 1)] > -100):
            final_episode = episode
            print(f"You solved it in {final_episode} episodes!")
            break
        
    # if final_episode doesn't exist set too max episodes.
    final_episode = final_episode if 'final_episode' in locals() else max_episodes

    test_reward = test(env, TrainNet)
    
    print(f"Test reward: {test_reward}, avgerage reward last 100 episodes: {avg_rewards}, num episodes: {final_episode}")
    env.close()



if __name__ == "__main__":
    main() 
   