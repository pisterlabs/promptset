from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import gym
import rospy
import random
import os
import datetime

import openai_ros_envs.crib_task_env

tf.enable_eager_execution()

model = tf.keras.Sequential([
  tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(8,)),  # input shape required
  tf.keras.layers.Dense(16, activation=tf.nn.relu),
  tf.keras.layers.Dense(7)
])


model_dir = "/home/linzhank/ros_ws/src/turtlebot_rl/scripts/"
today = datetime.datetime.today().strftime("%Y%m%d")
checkpoint_dir = os.path.join(model_dir, today)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model,
                           optimizer_step=tf.train.get_or_create_global_step())
root.restore(tf.train.latest_checkpoint(checkpoint_dir))

a = np.array([[random.randrange(4)]])
s = np.random.randn(1,7)
sa = np.concatenate((s,a),axis=1).astype(np.float32)

s = model(sa)
