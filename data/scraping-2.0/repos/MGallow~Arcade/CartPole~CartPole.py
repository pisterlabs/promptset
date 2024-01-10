'''CartPole-v0'''

'''Playing Cartpole-v0 using the deep-Q learning algorithm'''

# which game are we playing from OpenAI's gym?
ENV_NAME = "CartPole-v0"


# optional, load model and weights
loaded_model = "CartPole-v0-scratch_500000.json"
loaded_weights = "CartPole-v0-scratch_500000.h5"
model_name = ENV_NAME + "-original_model"
if loaded_model is not None:
    model_name = loaded_model
arch_name = model_name + ".json"

Internet = True  # connected to the internet (Slack)?
Watch = True  # watch the training?
Train = False  # train or just play?
Record = False  # create video from testing?
observe = 100  # how many steps to observe before training?
nb_steps = 500000  # number of steps
nb_test_episodes = 3  # number of test episodes
max_nb_steps = 50000  # either nb_test_episodes or max_nb_steps -- whatever is shorter
gamma = 0.99  # decay rate of past Observations
explore = 5000  # frames over which to anneal epsilon
final_epsilon = 0.01  # final value of epsilon
initial_epsilon = 0.5  # starting value of epsilon
memory_max = 500  # number of previous transitions to remember
batch = 50  # size of minibatch


print("Using Internet:", Internet)
print("Environment:", ENV_NAME)
print("Watching:", Watch)
print("Training:", Train)
print("Record:", Record)

# access locations file
exec(open("Locations.py").read())
print("locations successfully read.")


#import dependencies
import os
import sys
import random
import numpy as np
from collections import deque
import gym
from PIL import Image
from keras.models import Sequential, model_from_json
from keras.regularizers import l2
from keras.initializations import normal, uniform
from keras.optimizers import SGD, rmsprop, Adam
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense


# define various modules for reproducibility
def save_weights(which_model, filename):
    run = os.path.join(runs_dir, filename)
    which_model.save_weights(run, overwrite=True)
    print("weights saved as", filename)


def save_architecture(archname, string):
    arch = os.path.join(arch_dir, archname)
    open(arch, 'w').write(string)
    print("architecture saved as", archname)


# Get the environment and extract the number of actions
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


# construct the model
def construct_model():
    if loaded_model is not None:
        # load previous architecture
        model_name = loaded_model
        model_location = os.path.join(arch_dir, loaded_model)
        model = model_from_json(open(model_location).read())
        if loaded_weights is not None:
            # load previous weights
            weights_location = os.path.join(runs_dir, loaded_weights)
            model.load_weights(weights_location)

    else:
        # if we are not loading a model, create one
        model = Sequential()
        model.add(Flatten(input_shape=(1,) +
                          env.observation_space.shape))  # (1, 4)
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        json_string = model.to_json()
        save_architecture(arch_name, json_string)
    print("model constructed.")
    return model


# train the model
def train_model():
    # initialize values
    memory = deque()
    t = 1
    episode = 1
    status = "Observing"
    total_reward = 0
    rewards = []
    epsilon = initial_epsilon
    x_t = env.reset()
    x_t = x_t.reshape(1, 1, x_t.shape[0])

    # train the model
    while t < nb_steps:
        if Watch:
            env.render(mode='human')
        if random.random() <= epsilon:
            print("----------Random Action----------")
            action = env.action_space.sample()
        else:
            # Get the prediction of the Q-function
            q = model.predict(x_t)
            # choose action with highest prediction of the Q-function
            action = np.argmax(q)

        # run the selected action and observed next state and reward
        x_t1, reward, done, info = env.step(action)

        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0])
        total_reward += reward

        # store the transition in D
        memory.append((x_t, action, reward, x_t1, done))
        if len(memory) > memory_max:
            memory.popleft()

        x_t = x_t1

        # reduce epsilon gradually
        if epsilon > final_epsilon:
            epsilon -= (initial_epsilon - final_epsilon) / explore

        if done:

            rewards.append(total_reward)
            print("********** EPISODE ENDED ***********")
            print("TOTAL REWARD:", total_reward,
                  "/EPISODE AVERAGE:", np.mean(rewards))

            # train on minibatches. We only train after observation period is
            # over so that there are sufficient number of frames to sample from
            if t > observe:
                # sample a minibatch to train on
                minibatch = random.sample(memory, batch)

                inputs = np.zeros(
                    (batch, x_t1.shape[1], x_t1.shape[2]))  # 50, 1, 4
                targets = np.zeros((batch, nb_actions))  # 50, 2

                # Now we do the experience replay
                for i in range(0, len(minibatch)):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    done = minibatch[i][4]
                    inputs[i:i + 1] = state_t
                    targets[i] = model.predict(state_t)
                    Q_sa = model.predict(state_t1)
                    max_Q = np.max(Q_sa)

                    if done:
                        targets[i, action_t] = reward_t
                    else:
                        targets[i, action_t] = reward_t + gamma * max_Q

                loss = 0
                loss += model.train_on_batch(inputs, targets)

                print("Minibatch complete. Loss: ", loss)
                print("saving weights.")
                save_weights(model, model_name + ".h5")

            total_reward = 0
            episode += 1
            x_t = env.reset()
            x_t = x_t.reshape(1, 1, x_t.shape[0])

            if t > observe:
                status = "Training"

        print("TIMESTEP", t, "/ EPISODE", episode, "/ STATUS", status,
              "/ EPSILON", round(epsilon, 4), "/ ACTION", action, "/ REWARD", reward, "/ DONE", done)
        t = t + 1

    if Internet:
        # send message to slack, signaling the end of training
        slck = os.path.join(slack_dir, "Slack.py")
        exec(open(slck).read())
        print("message sent to Slack")


# test the model
def test_model():
    # initialize values
    t = 1
    episode = 1
    status = "Testing"
    total_reward = 0
    rewards = []
    x_t = env.reset()
    x_t = x_t.reshape(1, 1, x_t.shape[0])

    # train the model
    while t < max_nb_steps and episode < nb_test_episodes:
        if Watch:
            env.render(mode='human')

        # Get the prediction of the Q-function
        q = model.predict(x_t)
        # choose action with highest prediction of the Q-function
        action = np.argmax(q)

        # run the selected action and observed next state and reward
        x_t1, reward, done, info = env.step(action)

        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0])
        total_reward += reward

        x_t = x_t1

        if done:
            rewards.append(total_reward)
            print("********** EPISODE ENDED ***********")
            print("TOTAL REWARD:", total_reward,
                  "/EPISODE AVERAGE:", np.mean(rewards))

            total_reward = 0
            episode += 1
            x_t = env.reset()
            x_t = x_t.reshape(1, 1, x_t.shape[0])

        print("TIMESTEP", t, "/ EPISODE", episode, "/ STATUS", status,
              "/ ACTION", action, "/ REWARD", reward, "/ DONE", done)
        t = t + 1


# construct model
model = construct_model()
# compile the model
model.compile(loss='mse', optimizer='rmsprop')


if Train:
    train_model()
if not Train:
    test_model()
