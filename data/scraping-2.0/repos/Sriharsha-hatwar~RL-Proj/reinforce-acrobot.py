import time
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch.nn.functional as F
from itertools import product
import json


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

class AcrobotAgent():
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
        policy_function_path = path + self.env.spec.id + "-" + "policy_function.pt"
        value_function_path = path + self.env.spec.id + "-"  + "value_function.pt"
        policy_model = PolicyModel(self.env.observation_space.shape[0], self.env.action_space.n, 256, 128, 64).to(self.device)
        value_model = ValueModel(self.env.observation_space.shape[0], 1, 256, 128, 64).to(self.device)
        policy_model.load_state_dict(torch.load(policy_function_path))
        value_model.load_state_dict(torch.load(value_function_path))
        self.policy_function = policy_model
        self.value_function_approxmatior = value_model
    
    def test_agent_in_env(self, no_episodes):
        # Test the agent in the environment with the saved states. 
        env = gym.make("Acrobot-v1", render_mode="human")
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
    no_episodes = 5000 # The number of episodes to train for.
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

def stopping_criteria_reached(epi_rewards, agent, hyper_param_done, value_appr_losses, policy_losses):
    # This contains a tuple of episode number and the total reward.
    # So this will contain another way of training, check for
    # Check for last 20 episodes where the mean reward is greater than 230 and 
    # minimum reward is greater than 200 and then save the model weights.
    if len(epi_rewards) > 10 : 
        rewards = [reward for _, reward in epi_rewards[-10:]]
        min_reward = np.min(rewards)
        if min_reward > -90:
            if not hyper_param_done:
                # Also, save everything, like loss and such, so get that ffrom the training loop.
                print("You know what screw this, I am saving you.")
                agent.save_model("./temporary_models/")
                np.save(f"acrobot_logs/pre-saving-acrobot-value-loss.npy", value_appr_losses)
                np.save(f"acrobot_logs/pre-saving-acrobot-policy-loss.npy", policy_losses)
                np.save(f"acrobot_logs/pre-saving-acrobot-epi-rewards.npy", epi_rewards)
            else:
                pass
            return True
        
        else:
            return False 
    else:
        return False

def train_reinforce(env_name, hyper_params, device, hyper_param_done):
    # Initialize the agent 
    env = gym.make(env_name)
    if hyper_param_done:
        pass
    else:
        set_seed(env, hyper_params["seed"])
    agent = None
    if env_name== "Acrobot-v1":
        # Initialize the agent.
        agent = AcrobotAgent(env, hyper_params["alpha_policy"], hyper_params["alpha_weights"], hyper_params["gamma"], device, False)
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

        if stopping_criteria_reached(epi_rewards, agent, hyper_param_done, value_appr_losses, policy_losses):
            print("The last 10 episodes have a minimum reward greater than -100, so bail out.")
            return np.mean([reward for _, reward in epi_rewards[-10:]]), epi_rewards, value_appr_losses, policy_losses, agent

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
                # Print the metrics every alternate episode.
                if episode % 10 == 0:
                    # Can average the loss over the last 10 episodes.
                    val_appr_loss = np.mean([loss for _, loss in value_appr_losses[-10:]])
                    policy_loss = np.mean([loss for _, loss in policy_losses[-10:]])
                    average_reward = np.mean([reward for _, reward in epi_rewards[-10:]])
                    print(f"Episode : {episode}, Total Reward : {average_reward}, Value Approximator Loss : {val_appr_loss}, Policy Loss : {policy_loss}")
            len_episode_counter += 1
    # Save the model weights if the last 10 epi_rewards has majority of them going over 190.

    return np.mean([reward for _, reward in epi_rewards[-10:]]), epi_rewards, value_appr_losses, policy_losses, agent
        
def hyper_parameter_search():
    # TODO : Implement this using ray tune.
    pass

def run_simulation_with_best_hyperparams():
    env_name = "Acrobot-v1"
    env = gym.make(env_name)
    agent = AcrobotAgent(env, None, None, None, device, True)
    agent.load_model("./temporary_models/",)
    agent.test_agent_in_env(5)

def hyper_param_tuning(env_name, device):
    # TODO : Implement this.
    # What are the parameters to tune?
    # The learning rate for the policy network.
    # The learning rate for the weights network.
    # The Seed. 
    # The number of steps.

    hyperparameter_grid = {
        "num_episodes" : [3000],
        "alpha_policy" : [1e-3, 1e-2, 1e-4], # 1e-2, 1e-3, 1e-4
        "alpha_weights" : [1e-2, 0.1, 1e-3], # , 0.1, 1e-3, 0.01
        "seed" : [42],  # 43, 44
        "gamma" : [0.99], # 0.95
        "reward_scale" : [1]
    }

    hyperparameter_combinations = list(product(*hyperparameter_grid.values()))
    reward_hyperparameter_list = []
    for params in hyperparameter_combinations:
        hyperparameter_dict = dict(zip(hyperparameter_grid.keys(), params))
        print("Training for Hyperparameter dict : ", hyperparameter_dict)
        time_start = time.time()
        mean_reward, _, _, _, _  = train_reinforce(env_name, hyperparameter_dict, device, False)
        time_end = time.time()
        reward_hyperparameter_list.append((hyperparameter_dict, mean_reward, time_end - time_start))
        #print("Time taken for training : ", time_end - time_start)

    # First sort the list according to the reward, if all rewards are the same, then sort according to the time taken.
    sorted_reward_hp_list = sorted(reward_hyperparameter_list, key=lambda x: (-x[1], x[2]))
    print("Sorted reward hyperparameter list : ", sorted_reward_hp_list)
    print("Reward hyperparameter list : ", sorted_reward_hp_list)
    print("Best hyperparameter combination : ", sorted_reward_hp_list[0][0])
    

    # Now use that to training the agent again and run the simulation 10 times! 
    print("Running the training with the best hyperparameter combination for 10 times with different seeds.")
    best_hyperparameter_dict = sorted_reward_hp_list[0][0]
    
    print("Saving the conf in .npy file.")
    np.save(f"acrobot_logs/cartpole-best_hyperparameter_dict.npy", best_hyperparameter_dict)
    # write dict to a file
    with open("cartpole_logs/cartpole-best_hyperparameter_dict.txt", "w") as file:
        file.write(json.dumps(best_hyperparameter_dict))

    best_reward = -500
    for i in range(10):
        print("Running the simulation for the ", i, "th time. ")
        best_hyperparameter_dict["seed"] = best_hyperparameter_dict["seed"] + i
        (mean_reward, 
        per_episode_reward, 
        value_approximation_losses, 
        policy_losses, agent) = train_reinforce(env_name, best_hyperparameter_dict, device, True)
        np.save(f"acrobot_logs/acrobot-mean_reward-{i}.npy", mean_reward)
        np.save(f"acrobot_logs/acrobot-per_episode_reward-{i}.npy", per_episode_reward)
        np.save(f"acrobot_logs/acrobot-value_approximation_losses-{i}.npy", value_approximation_losses)
        np.save(f"acrobot_logs/acrobot-policy_losses-{i}.npy", policy_losses)
        if mean_reward > best_reward:
            best_reward = mean_reward
            print("Best mean, so saving the model weights.")
            agent.save_model("./secondary_models/")
    # Now run the simulation.
    run_simulation_with_best_hyperparams()
    # Dump all the information, mean_reward, per_episode_reward, value_approximation_losses, policy_losses, step_per_each_episodes
    # into a .npy file.
    
    print("Dumped all the information into .npy files.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = "Acrobot-v1"
    ##train_reinforce(env_name, device)
    # test the agent.
    env = gym.make(env_name)
    agent = AcrobotAgent(env, None, None, None, device, True)
    agent.load_model("./temporary_models/")
    agent.test_agent_in_env(10)
    #hyper_param_tuning(env_name, device)

