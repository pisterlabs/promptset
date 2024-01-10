import gymnasium as gym
import numpy as np
import itertools
import matplotlib.pyplot as plt

from openai_gym.utils import parse_args

class CartPole:
    def __init__(self, config, render_mode=None):
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_steps = self.env.spec.max_episode_steps
        self.num_episodes = config['num_episodes']
        # load q-table if it exists, otherwise initialize to zeros
        self.on_policy = config['on_policy']
        self.bins = config['bins']
        try:
            self.q_table = np.load('q_table_on_policy.npy') if self.on_policy else np.load('q_table.npy')
        except FileNotFoundError:
            self.q_table = np.zeros(self.bins + [self.action_space.n])
        self.alpha = config['alpha'] # learning rate
        self.gamma = config['gamma'] # discount factor
        self.epsilon = config['epsilon'] # exploration rate
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.rewards = []
        self.episode_number = 0
        # self.alpha = 0.1 # learning rate
        # self.gamma = 0.9 # discount factor

    def decay_epsilon(self):
        # a number of things can be done here, depending on the problem
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon - 0.01) # much better performance

    def discretize_state(self, observation):
        # bucketing continuous values into discrete bins
        # somewhat arbitrarily starting with 10 bins
        cart_position, cart_velocity, pole_angle, pole_velocity = observation

        # np.digitize is 1-indexed, so subtract one to get 0-indexed
        bin_cart_position = np.digitize(cart_position, np.linspace(-2.4, 2.4, self.bins[0])) - 1
        bin_cart_velocity = np.digitize(cart_velocity, np.linspace(-3, 3, self.bins[1])) - 1
        bin_pole_angle = np.digitize(pole_angle, np.linspace(-0.2095, 0.2095, self.bins[2])) - 1
        bin_pole_velocity = np.digitize(pole_velocity, np.linspace(-3, 3, self.bins[3])) - 1
        return (int(bin_cart_position), int(bin_cart_velocity), int(bin_pole_angle), int(bin_pole_velocity))
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            # explore
            return self.action_space.sample()
        else:
            # exploit
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, next_action):
        if self.on_policy:
            # SARSA
            self.q_table[state][action] += self.alpha * (reward + self.gamma * self.q_table[next_state][next_action] - self.q_table[state][action])
        else:
            # Q-learning
            self.q_table[state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])
    
    def run(self):
        for i in range(self.num_episodes):
            self.episode_number += 1
            episode_reward = self.run_episode()
            self.rewards.append(episode_reward)
            self.decay_epsilon()
            print(f"Episode {self.episode_number} reward: {episode_reward}")
        self.env.close()

    def run_episode(self):
        observation, info = self.env.reset()
        state = self.discretize_state(observation)
        action = self.get_action(state)
        total_reward = 0
        for i in range(self.max_steps):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            next_state = self.discretize_state(observation)
            next_action = self.get_action(next_state)
            self.update_q_table(state, action, reward, next_state, next_action)

            if terminated or truncated:
                break
            state, action = next_state, next_action
        return total_reward

def run_experiment(config):
    cartpole = CartPole(config)
    cartpole.run()
    avg_reward = sum(cartpole.rewards[:100]) / 100
    return avg_reward

def plot_results(results):
    alphas = [config['alpha'] for config, _ in results]
    performances = [performance for _, performance in results]

    plt.figure(figsize=(10, 6))
    plt.scatter(alphas, performances, c='blue')
    plt.title('Hyperparameter Tuning Results')
    plt.xlabel('Alpha')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.show()

def hyperparameter_tuning():
    results = []
    alpha_values = [0.01, 0.1, 0.5]
    gamma_values = [0.8, 0.9, 0.99]
    epsilon_decay_values = [0.99, 0.995, 0.999]
    epsilon_min_values = [0.01, 0.05, 0.1]
    bins = [[10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30]]
    on_off_policy_values = [True, False]

    best_performance = -np.inf
    best_config = None

    for alpha, gamma, epsilon_decay, epsilon_min, bins in itertools.product(alpha_values, gamma_values, epsilon_decay_values, epsilon_min_values, bins):
        for on_policy in on_off_policy_values:
            config = {
                'num_episodes': 10000,
                'alpha': alpha,
                'gamma': gamma,
                'epsilon': 1.0,
                'epsilon_decay': epsilon_decay,
                'epsilon_min': epsilon_min,
                'on_policy': on_policy,
                'bins': bins
            }

            performance = run_experiment(config)
            if performance > best_performance:
                best_performance = performance
                best_config = config

            print(f"Tested config: {config}, performance: {performance}")
            results.append((config, performance))

    print(f"Best config: {best_config}, Best performance: {best_performance}")
    return results

if __name__ == '__main__':
    results = hyperparameter_tuning()
    print(results)
    plot_results(results)
    # num_episodes, render_mode = parse_args()
    # config = {
    #     'num_episodes': num_episodes,
    #     'on_policy': True,
    #     'alpha': 0.1,
    #     'gamma': 0.9,
    #     'epsilon_decay': 0.99,
    #     'epsilon_min': 0.01,
    #     'epsilon': 1.0,
    #     'bins': [30, 30, 30, 30]
    # }

    # cartpole = CartPole(config, render_mode=render_mode)
    # try:
    #     cartpole.run()
    # except KeyboardInterrupt:
    #     # write q-table to file if user interrupts
    #     pass
    # if config['on_policy']:
    #     np.save('q_table_on_policy.npy', cartpole.q_table)
    # else:
    #     np.save('q_table.npy', cartpole.q_table)