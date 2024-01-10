from dotenv import load_dotenv
import gymnasium as gym
import os
from openai import OpenAI
import numpy as np
import openai
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import base64
from PIL import Image
import requests
from time import sleep
import json
import re


load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
NUM_EPISODES = 100
DISCOUNT_FACTOR = 0.8

current_state_file = "current_state.png"
actions_dict = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}


def format_state_vector(state_one_hot):
    formatted_string = ""
    for i in range(0, 16, 4):
        row = ','.join(state_one_hot[i:i+4])
        formatted_string += '[' + row + ']' + "\n"
    return formatted_string

def find_single_number(input_string):
    numbers = re.findall(r'\b\d+\b', input_string)
    if len(numbers) != 1:
        raise ValueError("Error: More than one number found in the string.")
    return int(numbers[0])


def gpt_4_naive_prompt():
    model = "gpt-4-1106-preview"

    client = OpenAI(api_key=OPENAI_KEY)
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    total_rewards = []

    for episode in tqdm(range(NUM_EPISODES), desc="Training Episodes"):
        state = env.reset()[0]
        state_one_hot = np.zeros(env.observation_space.n)
        state_one_hot[state] = 1  # One-hot encode the state

        done = False
        episode_log_probs = []
        episode_rewards = []
        text_prompt = ("You are playing Frozen Lake. You need to answer in this format: 'The best choice is 2'. The action space is as follows 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP")

        messages=[{"role": "system", "content": text_prompt}]
        print(f"Episode {episode}")
        while not done:
            state_one_hot_string = ''.join(map(str, state_one_hot.astype(int)))

            state_one_hot_string = format_state_vector(state_one_hot_string)
            messages.append({"role": "user", "content": f"Current state vector:\n{state_one_hot_string}"})

            response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                )
            action = int(find_single_number(response.choices[0].message.content))

            next_state, reward, done, _, _ = env.step(action)
            messages.append({"role": "system", "content": f"You moved {actions_dict[action]}, and received a reward of {reward}"}) 


            next_state_one_hot = np.zeros(env.observation_space.n)
            next_state_one_hot[next_state] = 1

            episode_rewards.append(reward)

            state_one_hot = next_state_one_hot
            if reward == 1:
                done = True
                print("Goal reached!")
                break

        total_rewards.append(sum(episode_rewards))
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
    return total_rewards

def gpt_4(model: str = "gpt-4-1106-preview"):
    client = OpenAI(api_key=OPENAI_KEY)
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    
    total_rewards = []

    def format_state_vector(state_one_hot):
        formatted_string = ""
        for i in range(0, 16, 4):
            row = ','.join(state_one_hot[i:i+4])
            formatted_string += '[' + row + ']' + "\n"
        return formatted_string

    for episode in tqdm(range(NUM_EPISODES), desc="Training Episodes"):
        state = env.reset()[0]
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
        print(f"{text_prompt}")

        messages=[{"role": "system", "content": text_prompt}]
        print(f"Episode {episode}")
        while not done:
            state_one_hot_string = ''.join(map(str, state_one_hot.astype(int)))

            state_one_hot_string = format_state_vector(state_one_hot_string)
            messages.append({"role": "user", "content": f"Current state vector:\n{state_one_hot_string}"})

            response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                )
            action = int(response.choices[0].message.content[-1])

            next_state, reward, done, _, _ = env.step(action)
            messages.append({"role": "system", "content": f"You moved {actions_dict[action]}, and received a reward of {reward}"}) 


            next_state_one_hot = np.zeros(env.observation_space.n)
            next_state_one_hot[next_state] = 1

            episode_rewards.append(reward)

            state_one_hot = next_state_one_hot
            if reward == 1:
                done = True
                print("Goal reached!")
                break

        total_rewards.append(sum(episode_rewards))
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
    return total_rewards

def gpt_4V(model: str = "gpt-4-vision-preview"):
    current_state_file = "current_state.png"
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False,render_mode='rgb_array')
    
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_KEY}"
            }
    total_rewards = []

    def encode_image(image_rgb_array):
        im = Image.fromarray(image_rgb_array)
        im.save(current_state_file, format="PNG")
        with open(current_state_file, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    for episode in tqdm(range(NUM_EPISODES), desc="Training Episodes"):
        state = env.reset()[0]
        state_one_hot = np.zeros(env.observation_space.n)
        state_one_hot[state] = 1  # One-hot encode the state

        done = False
        episode_log_probs = []
        episode_rewards = []
        text_prompt = ("Game: Frozen Lake\n\n"
                "Action Space:\n"
                "0: Move LEFT\n"
                "1: Move DOWN\n"
                "2: Move RIGHT\n"
                "3: Move UP\n\n"
                "Directions:\n"
                "For a move to the right, respond with 'The best choice is 2' Any other response will break the system\n\n"
                "Now let's play the game\n")
        print(f"{text_prompt}")
        payload={"model": model,
                 "messages":[{"role": "system", "content": text_prompt}]}
        print(f"Episode {episode}")
        i=0
        while not done and i<16: #16 is the max number of moves, sometimes the agent gets stuck in a loop

            img = env.render()
            base64_image = encode_image(img)

            payload["messages"].append({"role": "user", "content": [{"type": "text","text":f"Play the next move"},
                                                                    {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                                                    ]})
            
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response = response.json()
            try:
                action = int(response["choices"][0]["message"]["content"][-1])
            except:
                #pretty print response
                import json
                print(json.dumps(response, indent=4, sort_keys=True))
                print(len(payload["messages"]))
                print("some mistake retrying this step")
                i+=1
                continue
            next_state, reward, done, _, _ = env.step(action)
            payload["messages"].append({"role": "system", "content": f"You moved {actions_dict[action]}, and received a reward of {reward}"}) 


            next_state_one_hot = np.zeros(env.observation_space.n)
            next_state_one_hot[next_state] = 1

            episode_rewards.append(reward)

            state_one_hot = next_state_one_hot
            if reward == 1:
                done = True
                print("Goal reached!")
                break
            i+=1
        if i==16:
            print("Too many moves")
            

        total_rewards.append(sum(episode_rewards))
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
    return total_rewards

def gpt_4V_plus_text(model = "gpt-4-vision-preview"):

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False,render_mode='rgb_array')
    
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_KEY}"
            }
    total_rewards = []

    def encode_image(image_rgb_array):
        im = Image.fromarray(image_rgb_array)
        im.save(current_state_file, format="PNG")
        with open(current_state_file, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def format_state_vector(state_one_hot):
        formatted_string = ""
        for i in range(0, 16, 4):
            row = ','.join(state_one_hot[i:i+4])
            formatted_string += '[' + row + ']' + "\n"
        return formatted_string


    for episode in tqdm(range(NUM_EPISODES), desc="Training Episodes"):
        state = env.reset()[0]
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
        payload={"model": model,
                 "messages":[{"role": "system", "content": text_prompt}]}
        print(f"Episode {episode}")
        i=0
        while not done and i<16:

            img = env.render()
            base64_image = encode_image(img)
            state_one_hot_string = ''.join(map(str, state_one_hot.astype(int)))

            state_one_hot_string = format_state_vector(state_one_hot_string)

            payload["messages"].append({"role": "user", "content": [{"type": "text","text":f"Play the next move"},
                                                                    {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                                                    ]})
            
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response = response.json()
            try:
                action = int(response["choices"][0]["message"]["content"][-1])
            except:
                print(json.dumps(response, indent=4, sort_keys=True))
                print(len(payload["messages"]))
                print("some mistake retrying this step")
                i+=1
                continue
            next_state, reward, done, _, _ = env.step(action)
            payload["messages"].append({"role": "system", "content": f"You moved {actions_dict[action]}, and received a reward of {reward}"}) 


            next_state_one_hot = np.zeros(env.observation_space.n)
            next_state_one_hot[next_state] = 1

            episode_rewards.append(reward)

            state_one_hot = next_state_one_hot
            if reward == 1:
                done = True
                print("Goal reached!")
                break
            i+=1
        if i==16:
            print("Too many moves")
            

        total_rewards.append(sum(episode_rewards))
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
    return total_rewards

