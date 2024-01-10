from dotenv import load_dotenv
import gymnasium as gym
import os
from openai import OpenAI
import numpy as np
from tqdm import tqdm
import torch
import base64


load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
NUM_EPISODES = 100
DISCOUNT_FACTOR = 0.8
MODEL = "gpt-4-1106-preview"

client = OpenAI(api_key=OPENAI_KEY)
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False,render_mode='rgb_array')
actions_dict = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}


def format_state_vector(state_one_hot):
  formatted_string = ""
  for i in range(0, 16, 4):
      row = ','.join(state_one_hot[i:i+4])
      formatted_string += '[' + row + ']' + "\n"
  return formatted_string

def encode_image(image):
    return base64.b64encode(image).decode('utf-8')


for episode in tqdm(range(NUM_EPISODES), desc="Training Episodes"):
    state = env.reset()[0]
    img = encode_image(env.render())
    state_one_hot = np.zeros(env.observation_space.n)
    state_one_hot[state] = 1  # One-hot encode the state

    done = False
    episode_log_probs = []
    episode_rewards = []
    text_prompt = ("Game: Frozen Lake\n\n"
            "Objective: Navigate on a 4x4 grid from Start (S) to Goal (G) without falling into Holes (H).\n\n"
            "Grid Space:\n"
            "[['S', 'F', 'F', 'F'],\n"
            " ['F', 'H', 'F', 'H'],\n"
            " ['F', 'F', 'F', 'H'],\n"
            " ['H', 'F', 'F', 'G']]\n\n"
            "'S' represents the game-starting position (character), 'H' represents a hole, 'F' represents the frozen lake surface (walkable path), and 'G' represents the goal (chest or box)\n\n"
            "YOU HAVE TO AVOID THE HOLES,H! THE ONLY WAY TO WIN IS TO REACH THE GOAL, G!\n"
            "Action Space:\n"
            "0: Move LEFT\n"
            "1: Move DOWN\n"
            "2: Move RIGHT\n"
            "3: Move UP\n\n"
            "Directions:\n"
            "Please note, movements are relative to the grid's orientation. You will fail if you fall off the map. Meaning you can't move outside of the boundaries\n\n"
            "Instructions:\n"
            "You will be provided with textual descriptions of the game's current state. Your response should be the numeric code for the next action you choose to take.\n\n"
            "Example:\n"
            "For a move to the right, respond with 'The best choice is 2' Any other response will break the system\n\n"
            "Now let's play the game\n")
    
    messages=[{"role": "system", "content": text_prompt}]
    print(f"Episode {episode}")
    while not done:
        state_one_hot_string = ''.join(map(str, state_one_hot.astype(int)))

        state_one_hot_string = format_state_vector(state_one_hot_string)
        print(f"Current state vector\n{state_one_hot_string}")
        full_prompt = text_prompt + state_one_hot_string
        messages.append({"role": "user", "content": f"Current state vector:\n{state_one_hot_string}"})
        response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
            )
        print(f"GPT's response is {response.choices[0].message.content}")
        action = int(response.choices[0].message.content[-1])

        next_state, reward, done, _, _ = env.step(action)
        
        messages.append({"role": "system", "content": [
                    {"type": "text", "text": f"You moved {actions_dict[action]}, and received a reward of {reward}"},
                    {
                     "type": "image_url",
                     "image_url": {
                     "url": f"data:image/jpeg;base64,{img}"
                        }
                    }
                ]
            } 
        ) 

        img = encode_image(env.render())

        next_state_one_hot = np.zeros(env.observation_space.n)
        next_state_one_hot[next_state] = 1

        episode_rewards.append(reward)

        state_one_hot = next_state_one_hot
        if reward == 1:
            done = True
            print("Goal reached!")
            break
        print(messages[1:])
    # Compute returns
    G = 0
    returns = []
    for reward in reversed(episode_rewards):
        G = reward + DISCOUNT_FACTOR * G
        returns.insert(0, G)  # Insert the return at the beginning of the list

    # Reverse the returns list so that it corresponds to the order of the episode_log_probs
    returns = returns[::-1]
    if episode % 10 == 0:
        print(f"Episode: {episode}, Total Reward: {sum(episode_rewards)}")

env.close()
