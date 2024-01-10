from __future__ import print_function, absolute_import, division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import gym
import rospy
import random
import os
import time
import datetime
import matplotlib.pyplot as plt

import openai_ros_envs.crib_task_env
import utils

tf.enable_eager_execution()


def loss(model, x, y):
  y_ = model(x)
  return tf.losses.mean_squared_error(labels=y, predictions=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

if __name__ == "__main__":
  rospy.init_node('env_test', anonymous=True, log_level=rospy.WARN)    
  env = gym.make('TurtlebotCrib-v0')
  
  rospy.loginfo("Gazebo gym environment set")
  # set parameters
  num_actions = env.action_space.n
  num_states = env.observation_space.shape[0]
  batch_size = 256
  
  stacs_memory = []
  nextstates_memory = []
  # setup model
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(num_states+1,)),  # input shape required
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(num_states)
  ])
  # set training parameters
  num_epochs = 128
  num_sample_steps = 32
  num_episodes = 16
  num_steps = 128

  # Random Sampling
  rs_start = time.time()
  for i in range(num_epochs):
    print("Sampling: {:03d}".format(i+1))
    state, info = env.reset()
    done = False
    state = state.astype(np.float32)
    for j in range(num_sample_steps):
      action = random.randrange(num_actions)
      next_state, _, done, info = env.step(action)
      print("Sampling {}, Step: {}, current_position: {}, goal_position: {}, done: {}".format(
        i,
        j,
        info["current_position"],
        info["goal_position"],
        done
      ))
      next_state = next_state.astype(np.float32)
      stac = np.concatenate((state, np.array([action]))).astype(np.float32)
      stacs_memory.append(stac)
      nextstates_memory.append(next_state.astype(np.float32))
      state = next_state
  rs_end = time.time()
  print("Random sampling takes: {:.4f}".format(rs_end-rs_start))
      
  # Train random sampled dataset
  dataset = utils.create_dataset(
    input_features=np.array(stacs_memory),
    output_labels=np.array(nextstates_memory),
    batch_size=batch_size,
    num_epochs=num_epochs
  )
  # for epoch in range(num_epochs):
  #   for i, (x, y) in enumerate(dataset):
  #     print("epoch: {:03d}, iter: {:03d}".format(epoch, i))
  #     print()
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  global_step = tf.train.get_or_create_global_step()
  loss_value, grads = grad(
    model,
    np.array(stacs_memory),
    np.array(nextstates_memory)
  )

  # train random samples
  rst_start = time.time()
  for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    for i, (x,y) in enumerate(dataset):
      batch_start = time.time()
      # optimize model
      loss_value, grads = grad(model, x, y)
      optimizer.apply_gradients(
      zip(grads, model.variables),
        global_step
      )
      # track progress
      epoch_loss_avg(loss_value)  # add current batch loss
      # log training
      print("Epoch {:03d}: Iteration: {:03d}, Loss: {:.3f}".format(epoch, i, epoch_loss_avg.result()))
      batch_end = time.time()
      print("Batch {} training takes: {:.4f}".format(i, batch_end-batch_start))
  rst_end = time.time()
  print("Random samples training takes {:.4f}".format(rst_end-rst_start))

  # Test model accuracy
  error_storage = np.zeros((num_episodes, num_steps))
  for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    state = state.astype(np.float32)
    for step in range(num_steps):
      action = random.randrange(num_actions)
      next_state, _, done, info = env.step(action)
      print("Sampling {}, Step: {}, current_position: {}, goal_position: {}, done: {}".format(
        i,
        j,
        info["current_position"],
        info["goal_position"],
        done
      ))
      next_state = next_state.astype(np.float32)
      stac = np.concatenate((state, np.array([action]))).astype(np.float32)
      predict = model(np.array([stac])).numpy()[0]
      stacs_memory.append(stac)
      nextstates_memory.append(next_state.astype(np.float32))
      state = next_state
      error_storage[episode, step] = np.linalg.norm(predict-next_state)
    

