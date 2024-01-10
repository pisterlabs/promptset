import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch.nn.functional as F


# We will evaluate the REINFORCE algorithm on the MountainCar-v0 environment
# from OpenAI Gym. The goal of the agent is to drive a car up a hill. However,
# the car is under-powered and cannot drive directly up the hill. Instead, it
# must drive back and forth to build up momentum. The agent receives a reward
# of -1 for each time step, until it reaches the goal at the top of the hill.
# The agent receives a reward of 0 if it reaches the goal. The agent receives
# a reward of -100 if it falls off the hill. The agent receives an observation
# of its position and velocity. The agent can choose to drive left, right, or
# not at all.

class PolicyModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size_one, hidden_size_two, hidden_size_three):
        # Here the input size is the size of the state space.
        # The output size is the size of the action space.
        super(PolicyModel, self).__init__()
        # Can experiment with different architectures.
        # TODO : Probably have a list of decreasing sizes and use that list while instantiating.
        self.linear_one = nn.Linear(input_size, hidden_size_one)
        self.linear_two = nn.Linear(hidden_size_one, hidden_size_two)
        self.linear_three = nn.Linear(hidden_size_two, hidden_size_three)
        self.linear_four = nn.Linear(hidden_size_three, output_size)
        self.relu = nn.ReLU()
        # To take the softmax over all the action values for a particular state, 
        # so taking the softmax over all the columns of a particular row.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.linear_one(x))
        x = self.relu(self.linear_two(x))
        x = self.relu(self.linear_three(x))
        x = self.softmax(self.linear_four(x))
        return x

class ValueModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size_one, hidden_size_two, hidden_size_three):
        # Can experiment with different architectures.
        # Input is the state space size.
        # Output is the value of the state.
        super(ValueModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear_one = nn.Linear(input_size, hidden_size_one)
        self.linear_two = nn.Linear(hidden_size_one, hidden_size_two)
        self.linear_three = nn.Linear(hidden_size_two, hidden_size_three)
        self.final_layer = nn.Linear(hidden_size_three, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear_one(x))
        x = self.relu(self.linear_two(x))
        x = self.relu(self.linear_three(x))
        x = self.final_layer(x)
        return x

class LunarLanderAgent():
    def __init__(self, env, alpha_policy_lr, alpha_value_lr, gamma, device, test_mode) -> None:
        self.env = env
        if not test_mode:
            self.value_function_approxmatior = ValueModel(env.observation_space.shape[0], 1, 256, 128, 64).to(device)
            self.policy_function = PolicyModel(env.observation_space.shape[0], env.action_space.n, 256 ,128, 64).to(device)
            self.value_function_optimizer = torch.optim.Adam(self.value_function_approxmatior.parameters(), lr=alpha_value_lr)
            self.policy_optimizer = torch.optim.Adam(self.policy_function.parameters(), lr=alpha_policy_lr)
        self.discount_factor = gamma
        self.action_size = env.action_space.n
        self.device = device

    def train(self, states, actions, rewards): 
        # Conver the list of rewards to a return.
        G = 0.0
        returns = [0.0] * len(rewards)
        for index, reward in enumerate(rewards[::-1]):
            G = G * self.discount_factor + reward
            returns[len(rewards) - index - 1] = G
            
        # this contains the returns, push those returns to be a tensor.
        returns = torch.FloatTensor(returns).to(self.device)
        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)

        
        # Calculate the value function approximator loss.
        f_values = self.value_function_approxmatior(states).squeeze()
        f_values_seperate = self.value_function_approxmatior(states).squeeze()
        #print("F values : ", f_values.shape)
        delta = returns - f_values
        action_probs = self.policy_function(states)
        # Check this line once again.
        #print("Action probs : ", action_probs.shape)
        #print("Actions : ", actions.shape)
        selected_prob_action = action_probs.gather(1, actions.view(-1,1))
        action_model_loss = torch.mean(-1.0 * torch.log(selected_prob_action) * delta)
        #print("Action model loss : ", action_model_loss)
        self.policy_optimizer.zero_grad()
        action_model_loss.backward()
        self.policy_optimizer.step()

        # Now for the value function approximator loss.
        value_function_loss = F.mse_loss(f_values_seperate, returns)
        #print("Value function loss : ", value_function_loss)
        self.value_function_optimizer.zero_grad()
        value_function_loss.backward()
   
        self.value_function_optimizer.step()
        #exit()
        return value_function_loss.detach().cpu().numpy(), action_model_loss.detach().cpu().numpy()
    
    def get_action(self, state):
        # This is like a eval function,  
        # Get the action from the policy_function
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            #print("State shape : ", state.shape)
            #print("State : ", state )
            action_probs = self.policy_function(state).squeeze().cpu().detach().numpy()
            # As these are stochastic, sample an action from the action_probs, 
            # choose the actions from the list of actions according to the action_probs.
            action = np.random.choice(np.arange(self.action_size), p=action_probs)
        return action
    
    def convert_to_return(self, rewards):
        # Convert the list of rewards to a return.
        # The return is the sum of the rewards.
        # Use the discount factor and the rewards to calculate the return.
        total_return = 0
        for i, reward in enumerate(rewards):
            total_return += (self.discount_factor ** i) * reward
        return sum(rewards)

    def save_model(self, path):
        # Save the model weights.
        print("Saving the model weights.")
        policy_function_path = path + self.env.spec.id + "-"  "policy_function.pt"
        value_function_path = path + self.env.spec.id + "-"  "value_function.pt"
        torch.save(self.policy_function.state_dict(), policy_function_path)
        torch.save(self.value_function_approxmatior.state_dict(), value_function_path)
        print("Model weights saved.")
        
    def load_model(self, path):
        # Load the model weights.
        policy_function_path = path + self.env.spec.id + "-"  "policy_function.pt"
        value_function_path = path + self.env.spec.id + "-"  "value_function.pt"
        policy_model = PolicyModel(self.env.observation_space.shape[0], self.env.action_space.n, 256, 128, 64).to(self.device)
        value_model = ValueModel(self.env.observation_space.shape[0], 1, 256, 128, 64).to(self.device)
        policy_model.load_state_dict(torch.load(policy_function_path))
        value_model.load_state_dict(torch.load(value_function_path))
        self.policy_function = policy_model
        self.value_function_approxmatior = value_model
    
    def test_agent_in_env(self, no_episodes):
        # Test the agent in the environment with the saved states. 
        env = gym.make("LunarLander-v2", render_mode="human")
        set_seed(env, 42)
        for episode in range(no_episodes):
            observation, info = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.get_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                if done or total_reward < -200  or total_reward > 190:
                    print("Terminated : ?  ", terminated)
                    print("Truncated : ? ", truncated)
                    print("Total reward in episode : ", episode, "is : ", total_reward)
                    break   
            #env.render()
        env.close()


def set_seed(env, seed):
    #env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_hyper_params():
    no_episodes = 30000 # The number of episodes to train for.
    gamma = 0.99 # The discount factor.
    # Is there any way to set the learning rate dynamically?
    alpha_policy = 1e-4 # The learning rate for the policy network.
    alpha_weights = 1e-4  # The learning rate for the weights network.
    reward_scale = 1 # TODO : This can be used to learn a better model by scaling the rewards.
    seed = 42 # The seed for the random number generators.
    return {"num_episodes" : no_episodes, 
            "gamma" : gamma, 
            "alpha_policy" : alpha_policy, 
            "alpha_weights" : alpha_weights, 
            "reward_scale" : reward_scale, 
            "seed" : seed}

def stopping_criteria_reached(epi_rewards, agent):
    # This contains a tuple of episode number and the total reward.
    # So this will contain another way of training, check for
    # Check for last 20 episodes where the mean reward is greater than 230 and 
    # minimum reward is greater than 200 and then save the model weights. 
    if len(epi_rewards) > 10:
        rewards = [reward for _, reward in epi_rewards[-10:]]
        mean_reward = np.mean(rewards)
        min_reward = np.min(rewards)
        if mean_reward > 230 and min_reward > 200:
            print("The rewards are : ", rewards)
            # Save the model weights now.
            agent.save_model("./lunar-lander-model/")
            return True
        else:
            return False


def end_of_episode_saving_criteria(epi_rewards):
    # if the last 10 episodes have a majority of them going over 190, then save the model weights.
    rewards = [reward for _, reward in epi_rewards[-10:]]
    # Check if the mean reward is greater than 200 or set to something lower
    if True or sum(rewards) > 1700:
        print("The rewards are : ", rewards)
        return True
    else:
        return False


def train_reinforce(env_name, device):
    best_last_10_rewards = -np.inf
    # Initialize the agent
    for i in range(1):
        print("Running the training loop for the ", i, "th time.")
        hyper_params = get_hyper_params() 
        env = gym.make(env_name)
        set_seed(env, hyper_params["seed"])
        agent = None
        if env_name== "LunarLander-v2":
            # Initialize the agent.
            agent = LunarLanderAgent(env, hyper_params["alpha_policy"], hyper_params["alpha_weights"], hyper_params["gamma"], device, False)
        elif env_name == "MountainCar-v0":
            # Initialize the agent.
            agent = None # TODO : Implement this.
        else:
            # For now.
            assert(agent is not None, "Agent is None.")
            pass
        epi_rewards = []
        value_appr_losses = []
        policy_losses = []
        len_episode_counter_list = []
        # Run the training loop for some number of episodes.
        for episode in range(hyper_params["num_episodes"]):
            # Reset the environment to get to a new starting state.
            state, _ = env.reset()
            # For each episode, maintain a list of states, actions, and rewards.
            states = []
            actions = []
            rewards = []
            len_episode_counter = 0
            # Run the episode until termination.
            done = False
            # Stop the process if the return / reward is greater than 200,
            # Reimplement the stopping criteria if its return.

            if stopping_criteria_reached(epi_rewards, agent):
                print("Stopping criteria reached, the reward is greater than 200, for the last 10 episodes.")
                break

            while not done:
                # Get the action from the agent policy.
                action = agent.get_action(state)
                # Take the action in the environment.
                next_state, reward, terminated, truncated, _ = env.step(action)
                #print("Next state : ", next_state)
                # Add the state, action, and reward to the lists.
                states.append(state)
                actions.append(action)
                rewards.append(reward * hyper_params["reward_scale"])
                # Get the losses and other metrics that are needed to be plotted.

                # Update the state.
                state = next_state
                done = terminated or truncated
                len_episode_counter += 1
                if done:
                    # then batch train the agent with the states, actions, and rewards.
                    value_appr_loss, policy_loss = agent.train(states, actions, rewards)
                    # Convert to return : sum of all rewards.
                    # or have to convert to returns, check this with the paper.
                    total_reward = sum(rewards)
                    # Plot the metrics.
                    epi_rewards.append((episode, total_reward))
                    value_appr_losses.append((episode, value_appr_loss))
                    policy_losses.append((episode, policy_loss))
                    len_episode_counter_list.append((episode, len_episode_counter))
                    # Print the metrics every alternate episode.
                    if episode % 10 == 0:
                        # Can average the loss over the last 10 episodes.
                        val_appr_loss = np.mean([loss for _, loss in value_appr_losses[-10:]])
                        policy_loss = np.mean([loss for _, loss in policy_losses[-10:]])
                        average_reward = np.mean([reward for _, reward in epi_rewards[-10:]])
                        print(f"Episode : {episode}, Total Reward : {average_reward}, Value Approximator Loss : {val_appr_loss}, Policy Loss : {policy_loss}")
            
        # Save the model weights if the last 10 epi_rewards has majority of them going over 190.
        np.save(f"lunar_lander_logs/temp-lunar-per_episode_reward-{i}.npy", epi_rewards)
        np.save(f"lunar_lander_logs/temp-lunar-per_episode_step_count-{i}.npy", len_episode_counter_list)
        average_reward = np.mean([reward for _, reward in epi_rewards[-10:]])
        if end_of_episode_saving_criteria(epi_rewards) and best_last_10_rewards < average_reward:
            best_last_10_rewards = average_reward
            print("Saving the model weights as mean is greater than previous best mean.")
            agent.save_model("./lunar-lander-model/")
        else:
            print("Model weights have not been saved...")
        
def hyper_parameter_search():
    # TODO : Implement this using ray tune.
    pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = "LunarLander-v2"
    #train_reinforce(env_name, device)
    # test the agent.
    env = gym.make(env_name)
    agent = LunarLanderAgent(env, None, None, None, device, True)
    agent.load_model("./lunar-lander-model/")
    agent.test_agent_in_env(10)

