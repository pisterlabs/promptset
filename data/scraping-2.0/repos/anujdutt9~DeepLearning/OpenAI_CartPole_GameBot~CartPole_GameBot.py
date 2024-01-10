# Game Bot for Playing the CartPole game from OpenAI

import sys
import os
import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

# Set the Learning Rate
learningRate = 1e-3
# Set the Game Environment
env = gym.make('CartPole-v0')
# Reset the Game Environment Initially
env.reset()

goal_steps = 500
# Train from all random games with a score > 50
score_requirement = 50
initial_games = 10000


# Make Random Games
# Illustrate how the random games look like
def random_games():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            # See what is happening in the Game
            env.render()
            # Takes a random action in environment to see how the Game responds
            action = env.action_space.sample()
            # Observation: array of data from game; mostly Pixel data; For this game it is Pole position, cart position
            # Reward: 1 or 0
            # Done: Game over or not
            # Info: Information
            observation, reward, done, info = env.step(action)
            if done:
                break


# Training to play the Game
# Create the Training Data
def initial_population():
    # Observation in the Move made
    # Append to training_data if score > 50
    training_data = []
    scores = []
    accepted_scores = []

    # Play the Game for 10,000 frames
    for _ in range(initial_games):
        # Initialize the game score
        score = 0
        # Store the movements of the game in game_memory to see at end
        game_memory = []
        # To store the previous observations
        prev_observation = []
        for _ in range(goal_steps):
            # Generate 0's and 1's
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            # Save the previous observations to Game Memory
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            # Update Previous observations
            prev_observation = observation
            score += reward
            if done:
                break

        # Check if the score obtained is > the required score i.e > 50
        if score >= score_requirement:
            # If score > 50, it is accepted
            # Append it to accepted_score
            accepted_scores.append(score)
            # Check if the observation is a "1" or a "0"
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                training_data.append([data[0], output])

        # Done with the game, reset it
        env.reset()
        # Apend all the scores that we saw "0's" or "1's"
        scores.append(score)
    # Save the training data as numpy array
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    # Print out the Analysis of Data
    print('Average Accepted Score: ', mean(accepted_scores))
    print('Median Accepted Score: ',median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


# Create the Model
def neural_network_model(input_size):
    # Define NN input data size
    network = input_data(shape=[None, input_size, 1], name='Input')

    # Create 5 hidden layers for the NN
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8, name='Dropout')

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8, name='Dropout')

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8, name='Dropout')

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8, name='Dropout')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8, name='Dropout')

    # Output Layer
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='Targets')

    # model = tflearn.DNN(network, tensorboard_dir='log')
    model = tflearn.DNN(network)
    return model


# Train the Neural Network
def train_model(training_data, model=False):
    # Define the Features
    # training_data: [observations, output]
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]),1)
    # Labels
    y = [i[1] for i in training_data]
    # If model has not been defined earlier
    if not model:
        model = neural_network_model(input_size=len(X[0]))
    # Train the Model
    model.fit({'Input': X}, {'Targets': y}, n_epoch= 5, snapshot_step=500, show_metric=True, run_id='openai_learning')

    return model


training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for each_game in range(20):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
        choices.append(action)

        new_obs, reward, done, info = env.step(action)
        prev_obs = new_obs
        game_memory.append([new_obs, action])
        score += reward
        if done:
            break
    scores.append(score)

print('Average Score: ',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_requirement)
model.save('./TrainedModel/trained_model.model')

# -------------------------------- EOC ------------------------------