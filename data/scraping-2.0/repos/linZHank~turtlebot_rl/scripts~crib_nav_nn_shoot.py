#! /usr/bin/env python

"""
Model based control with shooting method example in turtlebot crib environment

Navigate towards preset goal

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://bair.berkeley.edu/blog/2017/11/30/model-based-rl/

"""
from __future__ import absolute_import, division, print_function

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
  # init node
  rospy.init_node("crib_nav_mpc", anonymous=True, log_level=rospy.DEBUG)
  # create env
  env_name = "TurtlebotCrib-v0"
  env = gym.make(env_name)
  rospy.loginfo("Gazebo gym environment set")
  main_start = time.time()
  # set parameters
  num_actions = env.action_space.n
  num_states = env.observation_space.shape[0]
  num_episodes = 128
  num_steps = 256
  num_sequences = 1024
  len_horizon = 1024 # number of time steps the controller considers
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
  num_epochs = 512
  num_iters = 32

  # Random Sampling
  rs_start = time.time()
  for i in range(num_epochs):
    print("Sampling: {:03d}".format(i+1))
    state, info = env.reset()
    done = False
    state = state.astype(np.float32)
    for j in range(num_iters):
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
  # create check point
  model_dir = "/home/linzhank/ros_ws/src/turtlebot_rl/scripts/"
  today = datetime.datetime.today().strftime("%Y%m%d")
  checkpoint_prefix = os.path.join(model_dir, today, "ckpt")
  if not os.path.exists(os.path.dirname(checkpoint_prefix)):
    try:
      os.makedirs(checkpoint_prefix)
    except OSError as exc: # Guard against race condition
      if exc.errno != errno.EEXIST:
        raise
  root = tf.train.Checkpoint(
    optimizer=optimizer,
    model=model,
    optimizer_step=global_step
  )
  root.save(file_prefix=checkpoint_prefix)
  # train random samples
  rst_start = time.time()
  for epoch in range(num_epochs/8):
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

  # Control with more samples
  reward_storage = []
  for episode in range(num_episodes):
    state, info = env.reset()
    state = state.astype(np.float32)
    goal = info["goal_position"]
    total_reward = 0
    done = False
    # compute control policies as long as sampling more
    for step in range(num_steps):
      action_sequences = utils.generate_action_sequence(
        num_sequences,
        len_horizon,
        num_actions
      )
      action = utils.shoot_action(
        model,
        action_sequences,
        state,
        goal
      )
      next_state, reward, done, info = env.step(action)
      next_state = next_state.astype(np.float32)
      stac = np.concatenate((state, np.array([action]))).astype(np.float32)
      stacs_memory.append(stac)
      nextstates_memory.append(next_state)
      total_reward += reward
      state = next_state
      print("Current position: {}, Goal position: {}, Reward: {:.4f}".format(
        info["current_position"],
        info["goal_position"],
        reward
      ))
      if done:
        break
    reward_storage.append(total_reward)
    # Train with samples at the end of every episode
    dataset = utils.create_dataset(
      np.array(stacs_memory),
      np.array(nextstates_memory),
      batch_size=batch_size,
      num_epochs=4
    )
    ep_start = time.time()
    for i, (x, y) in enumerate(dataset):
      loss_value, grads = grad(model, x, y)
      optimizer.apply_gradients(
        zip(grads, model.variables),
        global_step
      )
      print("Batch: {:04d}, Loss: {:.4f}".format(i, loss_value))
    ep_end=time.time()
    print("Episode {:04d} training takes {:.4f}".format(episode, ep_end-ep_start))

    main_end = time.time()
    print(
      "{:d} Random Samples was trained {:d} epochs",
      "\n{:d} Controlled Samples was trained {:d} epochs",
      "\nTotal execution time: {:.4f}".format(
        num_epochs*num_iters,
        num_epochs/4,
        num_episodes*num_steps,
        num_episodes*4,
        main_end-main_start
      )
    )
