import os
import openai
import requests

os.environ["OPENAI_API_KEY"] = "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"
 
openai.api_key = os.environ["OPENAI_API_KEY"]

# def llm(prompt, stop=["\n"]):
#     response = openai.Completion.create(
#       model="text-curie-001",
#       prompt=prompt,
#       temperature=0,
#       max_tokens=100,
#       top_p=1,
#       frequency_penalty=0.0,
#       presence_penalty=0.0,
#       stop=stop
#     )
#     return response["choices"][0]["text"]

def openai_chat(**kwargs):
    url = "http://10.234.32.86:8880/chatgpt"
    data = kwargs
    data["type"] = "chat"
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, json=data, headers=headers)
    response = response.json()

    return response["data"]

def llm(prompt, stop=["\n"]):
    response = openai_chat(
                        n=1,
                        messages=[{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": f"{prompt}"}],
                        stop=stop,
                        temperature=0,
                        max_tokens=100,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                    )
    # print(response)
    return response["choices"][0]["message"]["content"]

import wikienv, wrappers
env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="tempquestions")
env = wrappers.LoggingWrapper(env)

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

import json
import sys

folder = './prompts/'
# prompt_file = 'prompts_naive.json'
prompt_file = 'zxy_prompts.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_prompt = prompt_dict['timequestions']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_prompt

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        # print("#"*100, "Prompt")
        # print(prompt)
        # print("#"*100)

        # print("#"*100, "Thought Idx")
        # print(i)
        # print("#"*100)
        
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        
        # print("#"*100, "Thought_Action")
        # print(thought_action)
        # print("#"*100)
        
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()

        # print("#"*100, "Action", type(action))
        # print(action)
        # print("#"*100)
        # action = "Lookup[杨程]"



        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    # print("#"*100, "Prompt")
    # print(prompt)
    # print("#"*100)
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info

import random
import time
idxs = list(range(7405))
# random.Random(233).shuffle(idxs)

rs = []
infos = []
old_time = time.time()
for i in idxs[:5]:
    r, info = webthink(i, to_print=True)
    rs.append(info['em'])
    infos.append(info)
    print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
    print('-----------')
    print()