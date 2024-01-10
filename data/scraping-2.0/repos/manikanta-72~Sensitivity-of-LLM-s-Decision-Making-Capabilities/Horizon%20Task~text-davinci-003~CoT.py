import openai
import gym
import envs.bandits
import time
import pandas as pd
import numpy as np
import random

import os
import csv

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

num2words = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
env = gym.make('wilson2014horizon-v0')
engine = "text-davinci-003"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(12))
def act(text):
    openai.api_key = "YOUR API KEY"
    response = openai.Completion.create(
        engine = engine,
        prompt = text,
        max_tokens = 1000,
        temperature = 0.5,
    )
    return response.choices[0].text.strip()

def reset(actions, rewards):
    print(actions.shape)
    print(rewards.shape)
    played_machines = []
    observed_rewards = []
    for t in range(4):
        played_machines.append('J' if actions[t, 0] else 'F')
        observed_rewards.append(str(int(rewards[t, 0, 1].item())) if actions[t, 0] else str(int(rewards[t, 0, 0].item())))

    trials_left = 'one additional round' if rewards.shape[0] == 5 else 'six additional rounds'

    answer_format = "Answer the following question in less than 200 words and you should summarize your choice at the end as 'Machine:[machine_name]'\n"

    instructions = "You are going to a casino that owns two slot machines.\n"\
        "You earn money each time you play on one of these machines.\n\n"\
        "You have received the following amount of dollars when playing in the past: \n"

    history = "- Machine " + played_machines[0] + " delivered " + observed_rewards[0] + " dollars.\n"\
        "- Machine " + played_machines[1] + " delivered " + observed_rewards[1] + " dollars.\n"\
        "- Machine " + played_machines[2] + " delivered " + observed_rewards[2] + " dollars.\n"\
        "- Machine " + played_machines[3] + " delivered " + observed_rewards[3] + " dollars.\n"

    trials_left = "Your goal is to maximize the sum of received dollars within " +  trials_left + ".\n"
    question = "Q: Which machine do you choose?\n"\
        "A: Let's think step by step."

    return answer_format, instructions, history, trials_left, question

def parse(llm_response):
    print("llm_response:", llm_response)
    if 'Machine:' in llm_response:
        sep_idx = llm_response.find('Machine:')
        action = llm_response[sep_idx+len('Machine:'):].replace(" ","").replace(".","").replace("'","").replace('"',"")
        print("action: ", action)
    elif 'alternate' in llm_response or 'Alternate' in llm_response:
        if random.choice([True, False]):
            action = 'F'
        else:
            action = 'J'
    else: 
        if 'choose Machine F' in llm_response or 'should choose is Machine F' in llm_response:
            action = 'F'
        elif 'choose Machine J' in llm_response or 'should choose is Machine F' in llm_response:
            action = 'J'
    # randomly choose the option if the model is unable choose one
    if 'Undecided' in action or 'Split' in action or'Either' in action or 'Unknown' in action or 'combination' in action or 'Alternate' in action or 'JF' in action or 'Alternating' in action or 'F-J' in action or 'UNDECIDED' in action or 'random' in action or 'UNKOWN' in action or 'N/A' in action or 'Both' in action or 'either' in action or 'Combination' in action:
        if random.choice([True, False]):
            action = 'F'
        else:
            action = 'J'

    return action

def step(history, action, rewards, t):
    trials_left = env.rewards.shape[0] - 5 - t

    if trials_left > 1:
        trials_left = num2words[trials_left] + " additional rounds"
    else:
        trials_left = num2words[trials_left] + " additional round"

    trials_left_string = "Your goal is to maximize the sum of received dollars within " +  trials_left + ".\n"

    if action == "F":
        trial_reward = int(rewards[t + 4, 0, 0].item())
    elif action == "J":
        trial_reward = int(rewards[t + 4, 0, 1].item())
    else:
        return None

    history += "- Machine " + action + " delivered " + str(trial_reward) + " dollars.\n"

    return history, trials_left_string

num_tasks = 600

for task in range(num_tasks):
    actions = [None, None, None, None]
    env.reset()

    answer_format, instructions, history, trials_left, question = reset(env.action, env.rewards)

    for t in range(env.rewards.shape[0] - 4):
        prompt = instructions + history + "\n" + trials_left + question
        print(prompt)
        llm_response = act(prompt)
        print(llm_response)
        action = parse(llm_response)

        if action == 'F':
            action_to_append = 0
        elif action == 'J':
            action_to_append = 1
        else:
            action_to_append = None
        actions.append(action_to_append)
        print(action)
        history, trials_left = step(history, action, env.rewards, t)
        if prompt is None:
            break
        print('')

    print(actions)
    if prompt is not None:
        data = []
        for trial in range(env.rewards.shape[0]):
            action = actions[trial] if trial >= 4 else env.action[trial, 0].item()
            row = [task, trial, env.mean_reward[0, 0].item(), env.mean_reward[0, 1].item(), env.rewards[trial, 0, 0].item(),  env.rewards[trial, 0, 1].item(), action]
            data.append(row)
        df = pd.DataFrame(data, columns=['task', 'trial', 'mean0', 'mean1', 'reward0', 'reward1', 'choice'])
        print(df)
        df.to_csv('data/' + 'cot/temp_05' +'/experiment' + str(task) + '.csv')