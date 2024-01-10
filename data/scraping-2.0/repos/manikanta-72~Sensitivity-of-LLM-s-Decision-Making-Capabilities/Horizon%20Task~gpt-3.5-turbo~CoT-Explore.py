import openai
import gym
import envs.bandits
import time
import pandas as pd
import numpy as np
import os
import csv
import random

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

num2words = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7:'seven', 8:'eight', 9:'nine', 10:'ten'}
env = gym.make('wilson2014horizon-v0')
engine = "gpt-3.5-turbo"

@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(12))
def act(text):
    openai.api_key = "YOUR API KEY"
    response = openai.ChatCompletion.create(
        model = engine,
        messages = [{"role":"user", "content":text}],
        max_tokens = 500,
        temperature = 1.0,
    )
    return response.choices[0].message.content.strip()

def reset(actions, rewards):
    print(actions.shape)
    print(rewards.shape)
    played_machines = []
    observed_rewards = []
    for t in range(4):
        played_machines.append('J' if actions[t, 0] else 'F')
        observed_rewards.append(str(int(rewards[t, 0, 1].item())) if actions[t, 0] else str(int(rewards[t, 0, 0].item())))

    trials_left = 'one additional round' if rewards.shape[0] == 5 else 'six additional rounds'

    answer_format = "Answer the following question and summarize your choice at the end as 'Machine:[machine_name]'.\n\n"

    instructions = "You are going to a casino that owns two slot machines.\n"\
        "You earn money each time you play on one of these machines.\n\n"\
        "You have received the following amount of dollars when playing in the past: \n"

    history = "- In round one, you chose Machine " + played_machines[0] + " and received " + observed_rewards[0] + " dollars.\n"\
        "- In round two, you chose Machine " + played_machines[1] + " and received " + observed_rewards[1] + " dollars.\n"\
        "- In round three, you chose Machine " + played_machines[2] + " and received " + observed_rewards[2] + " dollars.\n"\
        "- In round four, you chose Machine " + played_machines[3] + " and received " + observed_rewards[3] + " dollars.\n"

    hints = "The following hints will help you make good decision: \n"\
        "1. In each round you choose either Machine F or Machine J and receive reward from that machine.\n"\
        "2. If you have more than 4 rounds left, you choose the machine with less number of past observations.\n"
    
    trials_left = "Your goal is to maximize the sum of received dollars within " +  trials_left + ".\n"
    
    question = "Q: Which machine do you choose for next round?\n"\
        "A: Let's think step by step."

    return answer_format, instructions, history, hints, trials_left, question

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
    print("t :", t)
    history += "- In round " + num2words[5+t] + ", you chose Machine " + action + " delivered " + str(trial_reward) + " dollars.\n"

    return history, trials_left_string

def parse(llm_response):
    if 'Machine:' in llm_response:
        sep_idx = llm_response.find('Machine:')
        action = llm_response[sep_idx+len('Machine:'):].replace(" ","").replace(".","")
    # randomly choose a machine if LLM is unable to choose
    elif 'alternate' in llm_response or 'Alternate' in llm_response or 'randomly pick' in llm_response or 'Randomly choose':
        if random.choice([True, False]):
            action = 'F'
        else:
            action = 'J'
    
    return action.replace(" ", "")[0]
    
num_tasks = 600

for task in range(num_tasks):
    actions = [None, None, None, None]
    env.reset()

    answer_format, instructions, history, hints, trials_left, question = reset(env.action, env.rewards)

    for t in range(env.rewards.shape[0] - 4):
        prompt = answer_format + instructions + history + "\n" + hints + "\n" + trials_left + question
        print("prompt:", prompt)
        
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

    #print(actions)
    if prompt is not None:
        data = []
        for trial in range(env.rewards.shape[0]):
            action = actions[trial] if trial >= 4 else env.action[trial, 0].item()
            row = [task, trial, env.mean_reward[0, 0].item(), env.mean_reward[0, 1].item(), env.rewards[trial, 0, 0].item(),  env.rewards[trial, 0, 1].item(), action]
            data.append(row)
        df = pd.DataFrame(data, columns=['task', 'trial', 'mean0', 'mean1', 'reward0', 'reward1', 'choice'])
        print(df)
        df.to_csv('data/' + 'cot_explore/temp_10' +'/experiment' + str(task) + '.csv')