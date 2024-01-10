import gym
import numpy as np
from numpy.random import randint
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser(description="generate data from openai gym's FrozenLake-v0")
parser.add_argument('--trials', type=int,
			help="number of trials to simulate; default 10",
            default=100000)
parser.add_argument('--steps', type=int,
			help="number of decision steps to simulate for each trial; default 10",
            default=10)


args = parser.parse_args()

display = False

outfile = f"crossing_traffic_gym_{args.trials}x{args.steps}.tsv"

env = gym.make("CrossingTraffic-v0",max_steps=args.steps)

out_array = []

num_actions = env.action_space.n
state_vars = len(env.observation_space.spaces)
for i in range(args.trials):
    _ = env.reset()
    if display:
        print("game: "+str(i))
        env.render()
    done = False
    t=0
    log = [i]
    while not done and t<args.steps:
        if t == 0:
            SID = 0
        observations = env._get_obs()
        action = env.action_space.sample()
        _, reward, done, __ = env.step(action)
        log += list(observations)+[action,reward]
        if display:
            print(f"t: {t},\nobservation: {observations},\naction: {action},\nreward: {reward}")
            env.render()
        t+=1
    if display:
        print("\ntotal reward for trial " + str(i+1) + ":\t"+str(total_reward))
        all_trails_reward += total_reward
        if i>0 and i%100 == 0:
            print("average_reward:\t" + str(all_trails_reward/i))
    out_array.append(log)

import csv
with open(outfile, 'w') as myfile:
	wr = csv.writer(myfile,delimiter='\t')
	for log in out_array:
			wr.writerow(log)
