# Attempt to play Chopper Command using OpenAI library
# There are 2 versions of the game
#
# 1. RAM as input (ChopperCommand-ram-v0)
#      RAM of Atari 2600 consists of 128 bytes
#      AI nets score higher using this as input
# 2. Screen images as input (ChopperCommand-v0)
#      RGB image, array of shape (210, 160, 3)
#
# Each action is repeatedly performed for k frames,
# with k being uniformly sampled from {2,3,4}
#
# It seems that the highest scores were made using DQN,
# but not many used policy gradient methods. I will
# attempt to use policy gradient.


# Import OpenAI gym and other needed libraries
import gym
import tensorflow as tf
import numpy as np
import random
# import math
import time

# def policy_gradient():
#   with tf.variable_scope("policy"):

# def value_gradient():
#   with tf.variable_scope("value"):

def cnn_model(x, bn_is_training):
  # Batch Norm HyperParameters
  # bn_is_training = True
  bn_scale = True

  # We will create the model for our CNN here
  # Input layer takes in 104x80x3 = 25200
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 104, 80, 3])
    # print(x_image)

  # Conv 3x3 box across 3 color channels into 32 features
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([3,3,3,32])
    b_conv1 = bias_variable([32])
    pre_bn_conv1 = conv2d(x_image, W_conv1) + b_conv1
    post_bn_conv1 = tf.contrib.layers.batch_norm(pre_bn_conv1, center = True, scale = bn_scale, is_training = bn_is_training, scope = 'bn1')
    h_conv1 = tf.nn.relu(post_bn_conv1)
    # print(tf.shape(h_conv1))
    # print(h_conv1) shows shape = (1,104,80,32)

  # Not sure if doing this right, but adding a 2nd 3x3 filter...?
  # W_conv1_2 = weight_variable([3,3,3,32])
  # b_conv1_2 = bias_variable([32])
  # pre_bn_conv1_2 = conv2d(x_image, W_conv1_2) + b_conv1_2
  # post_bn_conv1_2 = tf.contrib.layers.batch_norm(pre_bn_conv1_2, center = True, scale = bn_scale, is_training = bn_is_training, scope = 'bn1_2')
  # h_conv1_2 = tf.nn.relu(post_bn_conv1_2)

  # Now, combine these two tensors? Should they be combined?
  # Before or after maxpool?
  # h_conv1_combined = tf.concat([h_conv1, h_conv1_2], axis_to_combine = 3)

  # Max pool to half size (52x40)
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # 2nd conv, 3x3 box from 32 to 64 features
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3,3,32,64])
    b_conv2 = bias_variable([64])
    pre_bn_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
    post_bn_conv2 = tf.contrib.layers.batch_norm(pre_bn_conv2, center = True, scale = bn_scale, is_training = bn_is_training, scope = 'bn2')
    h_conv2 = tf.nn.relu(post_bn_conv2)

  # 2nd max pool, half size again (26x20)
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # 3rd conv, 3x3 box from 64 to 128 features
  with tf.name_scope('conv3'):
    W_conv3 = weight_variable([3,3,64,128])
    b_conv3 = bias_variable([128])
    pre_bn_conv3 = conv2d(h_pool2, W_conv3) + b_conv3
    post_bn_conv3 = tf.contrib.layers.batch_norm(pre_bn_conv3, center = True, scale = bn_scale, is_training = bn_is_training, scope = 'bn3')
    h_conv3 = tf.nn.relu(post_bn_conv3)

  # 3rd max pool, half size last time (13x10)
  with tf.name_scope('pool3'):
    h_pool3 = max_pool_2x2(h_conv3)
    print(h_pool3)

  # First fully connected layer, 13*10*128 = 16640 to 512 fully connected
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([13*10*128, 512])
    b_fc1 = bias_variable([512])
    # Flatten max pool to enter fully connected layer
    h_pool3_flat = tf.reshape(h_pool3, [-1, 13*10*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Fully connected from 512 to 6 (1 for each action possible)
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([512, 6])
    b_fc2 = bias_variable([6])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  return y_conv, keep_prob

def conv2d(x, W):
  # Return full stride 2d conv
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  # 2x2 max pool
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def weight_variable(shape):
  # initial = tf.truncated_normal(shape, stddev=0.1)
  # USE XAVIER INITIALIZATION
  initial = tf.contrib.layers.xavier_initializer()
  return tf.Variable(initial(shape))

def bias_variable(shape):
  # initial = tf.constant(0.1, shape=shape)
  # USE XAVIER INITIALIZATION
  initial = tf.contrib.layers.xavier_initializer()
  return tf.Variable(initial(shape))

def choose_action(moveprobs):
  # Feed in probability and return an action 
  # Actions: up, down, left, right, shoot, nothing
  #           2     5     4      3      1        0
  # Fix this return after neural network is created
  # return random.randint(1,6)
  # sample_uniform = np.random.uniform()
  # Sample uniform from tensor??? Tensor is shape (-1, 6)
  # STOCHASTIC APPROACH
  # Psuedocode:
  #   sample_uniform = np.random.uniform()
  #   cumulated_sum = 0
  #     for i in range(5)
  #       cumulated_sum = cumulated_sum + moveprobs[current,0]
  #       if sample_uniform < cumulated_sum:
  #         return i
  #     return 5
  # This should return uniform distribution sampling of softmax probability
  

def main():
  # Prepare Chopper Command as env variable
  # and start access to images
  env = gym.make('ChopperCommand-v0')
  observation = env.reset()

  # observation now holds unsigned 8 bit int array
  # with shape (210, 160, 3). Let's half this for
  # our neural network to allow easier processing
  # by taking every other pixel
  reduced_observation = observation[::2, ::2, :]
  # Remove odd number from first observation
  reduced_observation = reduced_observation[1:, :, :]
  # reduced_observation is now shape (104,80,3)
  # Confirm reduced observation shape
  print("Reduced observation shape: ", reduced_observation.shape)
  float_input = reduced_observation.astype(np.float32)
  # reduced_observation.view('<f4')
  sess = tf.InteractiveSession()
  y_conv, keep_prob = cnn_model(float_input, True)
  moveprobs = tf.nn.softmax(y_conv)
  sess.run(tf.global_variables_initializer())
  print("Keep_prob: ", keep_prob)
  print("Y_Conv: ", y_conv)

  # Choosing to keep colors since enemies have different
  # colors. We can now feed this into our CNN.
  # Blue planes and white helicoptors = enemies
  # Black trucks = friendly

  # Reshape our array into 4-D tensor to allow input to NN  
  # input_layer = tf.reshape(reduced_observation, [-1,105,80,3])
  # print("Input Layer shape: ", input_layer.shape)

  #env.render allows us to render the graphics
  while True:
    observation, reward, done, info = env.step(choose_action(moveprobs))
    # print(observation)
    print(reward)
    # info is an object from the dict class, holding an attribute
    # called ale.lives, returns number of lives we have left
    # print(info['ale.lives'])
    time.sleep(0.05)
    env.render()
    # env.step()
    if done:
      # Compute weight changes and backpropagate here
      # Then reset the environment for another run.
      env.reset()

if __name__ == "__main__":
  main()
