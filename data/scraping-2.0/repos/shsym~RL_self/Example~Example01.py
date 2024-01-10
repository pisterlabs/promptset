import gym
import numpy as np

env = gym.make('FrozenLake-v0')     #load the environment from openAI gym

## Q-Table learning algorithm
# initialize the Q-tables with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])     #number of tiles, number of possible types of movement
# parameters
gamma = .8
y = .95
n_episodes = 2000

rList = []  #reward counter list
for i in range (n_episodes):    #we have total n_episodes number of independent tries
                                # (if it is finished, then reset, restarted)
    state = env.reset()     #initial state
    rAll = 0    #reward of current episode
    d = False
    j = 0

    #main part of Q-learning
    while j<99:
        j += 1

        #Q table for the current state with noise
        cur_q = Q[state, :] + np.random.randn(1, env.action_space.n) * (1./(i+1))   #noise increases as id of episode
        action = np.argmax(cur_q)

        #get the next state and reward
        next_state, reward, d, _ = env.step(action)  #we do the action

        #update q table from the given reward
        #new Q-value = old Q-value + discount * new Q-value
        #new Q-value =  reward + stepsize * target(=max(Q[next state])) - old_resard(=Q[cur_state, action])
        Q[state, action] += gamma * (reward + y * np.max(Q[next_state, :]) - Q[state, action])
        #update cumulative reward
        rAll += reward
        state = next_state
    rList.append(rAll)  #tracking total reward
    if i % 100 == 0:
        print("Reward[%03d]: %.3e" % (i, rAll))
        print("Score over time: %.3e" % (sum(rList) / (i+1)))

print ("Score over time: " + str(sum(rList) / n_episodes))
print("Final Q-Table Values")
print(Q)
