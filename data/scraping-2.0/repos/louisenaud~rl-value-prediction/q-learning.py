"""
Project:    rl-value-prediction
File:       q-learning.py
Created by: louise
On:         01/05/18
At:         1:42 PM
"""
import gym
import numpy as np
from gym import wrappers
import tqdm


def softmax(Q, beta=1.0):
    """
    Softmax function with parameter beta
    :param Q:
    :param beta: float, >= 0
    :return:
    """
    assert beta >= 0.0
    q_tilde = Q - np.max(Q)
    factors = np.exp(beta * q_tilde)
    return factors / np.sum(factors)


def softmax_policy(current_state, Q, beta=1.0):
    """
    Policy derived from Q / softmax strategy.
    :param current_state: OpenAI observation
    :param Q: numpy array for Q function
    :param beta: float, >= 0
    :return: OpenAI gym action (int)
    """
    prob_a = softmax(Q[current_state, :], beta=beta)
    cumsum_a = np.cumsum(prob_a)
    return np.where(np.random.rand() < cumsum_a)[0][0]


def epsilon_greedy_policy(current_state, Q, epsilon=0.1):
    """
    Policy derived from Q / epsilon greedy strategy
    :param current_state: OpenAI observation
    :param Q: numpy array for Q function
    :param epsilon: float, 0 < eps < 1
    :return: OpenAI gym action (int)
    """
    # Select action that maximizes Q at the current state p=1-eps
    a = np.argmax(Q[current_state, :])
    if np.random.rand() < epsilon:  # if random sample is < eps, then select random action instead p = eps.
        a = np.random.randint(Q.shape[1])
    return a


def run_episode(env, Q, learning_rate, discount, episode, render=False, policy="epsilon_greedy", beta=1.0, eps=0.5):
    """

    :param env: OpenAI gym environment
    :param Q: array storing Q function
    :param learning_rate: float, alpha
    :param discount: float, gamma
    :param episode: int, episode #
    :param render: bool, render environment
    :return:
    """
    # Initialize state and action
    observation = env.reset()
    done = False
    # Episode reward
    t_reward = 0
    # Get maximum possible number of states from OpenAI environment
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    # For each step of episode
    for i in range(max_steps):
        if done:  # if agent is at the goal
            break
        if render:
            env.render()

        current_state = observation
        # Get action from Q
        # Select an action from current state using policy derived from Q
        if policy == 'softmax':
            action = softmax_policy(current_state, Q, beta=beta)
        elif policy == 'epsilon_greedy':
            action = epsilon_greedy_policy(current_state, Q, epsilon=eps)
        elif policy == 'random':
            action = np.argmax(Q[current_state, :] + np.random.randn(1, env.action_space.n) * (1. / (episode + 1)))
        else:
            raise ValueError("Invalid policy_type: {}".format(policy))
        # Take action, observe reward and next state
        next_state, reward, done, info = env.step(action)
        # Update global reward for episode
        t_reward += reward
        # Compute TD error
        delta_t = reward + discount * np.max(Q[next_state, :]) - Q[current_state, action]
        # Update Q function
        Q[current_state, action] += learning_rate * delta_t

    return Q, t_reward


def main(env, num_episodes):
    """
    Run Q-learning for all episodes.
    :param env: OpenAI gym environment.
    :param num_episodes: int > 0
    :return: numpy array, Q function
    """
    learning_rate = 0.8
    discount = 0.99
    # Initialize Q-function array
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    # For each episode
    for i in tqdm.tqdm(range(num_episodes)):
        Q, reward = run_episode(env, Q, learning_rate, discount, i, render=False)

    return Q


if __name__ == "__main__":
    num_episodes = 100000
    env = gym.make('FrozenLake-v0')
    env = wrappers.Monitor(env, '/tmp/FrozenLake-experiment-6', force=True)
    q = main(env, num_episodes)
    print(q)
