#!/usr/bin/env python
import rospy

import gym
import keras
import numpy as np
import random

from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque

# import our training environment
from openai_ros.task_envs.cartpole_stay_up import stay_up



class ReplayBuffer():

  def __init__(self, max_size):
    self.max_size = max_size
    self.transitions = deque()

  def add(self, observation, action, reward, observation2):
    if len(self.transitions) > self.max_size:
      self.transitions.popleft()
    self.transitions.append((observation, action, reward, observation2))

  def sample(self, count):
    return random.sample(self.transitions, count)

  def size(self):
    return len(self.transitions)

def get_q(model, observation, state_size):
  np_obs = np.reshape(observation, [-1, state_size])
  return model.predict(np_obs)

def train(model, observations, targets, actions_dim, state_size):

  np_obs = np.reshape(observations, [-1, state_size])
  np_targets = np.reshape(targets, [-1, actions_dim])

  #model.fit(np_obs, np_targets, epochs=1, verbose=0)
  model.fit(np_obs, np_targets, nb_epoch=1, verbose=0)

def predict(model, observation, state_size):
  np_obs = np.reshape(observation, [-1, state_size])
  return model.predict(np_obs)

def get_model(state_size, learning_rate):
  model = Sequential()
  model.add(Dense(16, input_shape=(state_size, ), activation='relu'))
  model.add(Dense(16, input_shape=(state_size,), activation='relu'))
  model.add(Dense(2, activation='linear'))

  model.compile(
    optimizer=Adam(lr=learning_rate),
    loss='mse',
    metrics=[],
  )

  return model

def update_action(action_model, target_model, sample_transitions, actions_dim, state_size, gamma):
  random.shuffle(sample_transitions)
  batch_observations = []
  batch_targets = []

  for sample_transition in sample_transitions:
    old_observation, action, reward, observation = sample_transition

    targets = np.reshape(get_q(action_model, old_observation, state_size), actions_dim)
    targets[action] = reward
    if observation is not None:
      predictions = predict(target_model, observation, state_size)
      new_action = np.argmax(predictions)
      targets[action] += gamma * predictions[0, new_action]

    batch_observations.append(old_observation)
    batch_targets.append(targets)

  train(action_model, batch_observations, batch_targets, actions_dim, state_size)

def main():
  
  
  state_size = rospy.get_param('/cartpole_v0/state_size')
  action_size = rospy.get_param('/cartpole_v0/n_actions')
  gamma = rospy.get_param('/cartpole_v0/gamma')
  batch_size = rospy.get_param('/cartpole_v0/batch_size')
  target_update_freq = rospy.get_param('/cartpole_v0/target_update_freq')
  initial_random_action = rospy.get_param('/cartpole_v0/initial_random_action')
  replay_memory_size = rospy.get_param('/cartpole_v0/replay_memory_size')
  episodes_training = rospy.get_param('/cartpole_v0/episodes_training')
  max_iterations = rospy.get_param('/cartpole_v0/max_iterations')
  epsilon_decay = rospy.get_param('/cartpole_v0/max_iterations')
  learning_rate = rospy.get_param('/cartpole_v0/learning_rate')
  done_episode_reward = rospy.get_param('/cartpole_v0/done_episode_reward')
  
  steps_until_reset = target_update_freq
  random_action_probability = initial_random_action

  # Initialize replay memory D to capacity N
  replay = ReplayBuffer(replay_memory_size)

  # Initialize action-value model with random weights
  action_model = get_model(state_size, learning_rate)

  env = gym.make('CartPoleStayUp-v0')
  env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

  for episode in range(episodes_training):
    observation = env.reset()

    for iteration in range(max_iterations):
      random_action_probability *= epsilon_decay
      random_action_probability = max(random_action_probability, 0.1)
      old_observation = observation

      # We dont support render in openai_ros
      """
      if episode % 1 == 0:
        env.render()
      """
      
      if np.random.random() < random_action_probability:
        action = np.random.choice(range(action_size))
      else:
        q_values = get_q(action_model, observation, state_size)
        action = np.argmax(q_values)

      observation, reward, done, info = env.step(action)
      

      if done:
        print 'Episode {}, iterations: {}'.format(
          episode,
          iteration
        )

        # print action_model.get_weights()
        # print target_model.get_weights()

        #print 'Game finished after {} iterations'.format(iteration)
        reward = done_episode_reward
        replay.add(old_observation, action, reward, None)
        
        break

      replay.add(old_observation, action, reward, observation)

      if replay.size() >= batch_size:
        sample_transitions = replay.sample(batch_size)
        update_action(action_model, action_model, sample_transitions, action_size, state_size, gamma)
        steps_until_reset -= 1



if __name__ == "__main__":
    rospy.init_node('cartpole_mbalunovic_algorithm', anonymous=True, log_level=rospy.FATAL)
    main()