from typing import List, Dict, Tuple, Set, Optional, Union
import os 
import json 
import time 

import yaml 
import openai 

from schema import GPT3_CACHE

def setup(config_file:str)-> None:
    with open(config_file, 'r') as f:
        yaml_configs = yaml.load(f, Loader=yaml.FullLoader)
        
    openai.api_key = yaml_configs['OPENAI_API_KEY']
    return 

def load_cache(gpt_cache_dir, seed: int=-1)-> None:
    global GPT3_CACHE
    if seed != -1:
        gpt_cache_file = os.path.join(gpt_cache_dir, f'gpt_cache_{seed}.json')
    else:
        gpt_cache_file = os.path.join(gpt_cache_dir, f'gpt_cache.json')
    if os.path.exists(gpt_cache_file):
        with open(gpt_cache_file) as f:
            GPT3_CACHE = json.load(f)
    return


def save_cache(gpt_cache_dir, seed: int=-1)-> None:
    global GPT3_CACHE
    if seed != -1:
        gpt_cache_file = os.path.join(gpt_cache_dir, f'gpt_cache_{seed}.json')
    else:
        gpt_cache_file = os.path.join(gpt_cache_dir, f'gpt_cache.json')
    
    print(len(GPT3_CACHE))
    with open(gpt_cache_file,'w') as f:
        json.dump(GPT3_CACHE, f, indent=2)

    return 

def get_gpt3_response(prompt:str, use_cache:bool=True)->str:
    global GPT3_CACHE
    if use_cache and prompt in GPT3_CACHE:
        return GPT3_CACHE[prompt]['text']
    else:
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens=256,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                )
        except openai.OpenAIError:
            time.sleep(30)
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens=256,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                )

        GPT3_CACHE[prompt] = response['choices'][0]
        return response['choices'][0]['text']

def get_gpt3_score(prompt: str, use_cache: bool=True)-> Tuple[str, List[Dict[str, float]]]:
    global GPT3_CACHE
    if use_cache and prompt in GPT3_CACHE:
        predicted_text = GPT3_CACHE[prompt]['text']
        predicted_logprob = GPT3_CACHE[prompt]['logprobs']['top_logprobs'] # type: List[Dict[str, float]]
        return predicted_text, predicted_logprob

    else:
        try:
            response = openai.Completion.create(
                engine='text-davinci-003',
                prompt=prompt, 
                temperature=0.7,
                max_tokens=4, 
                top_p=0.95,
                frequency_penalty=0, 
                presence_penalty=0,
                logprobs=5
            )
        except openai.OpenAIError:
            time.sleep(10)
            response = openai.Completion.create(
                engine='text-davinci-003',
                prompt=prompt, 
                temperature=0.7,
                max_tokens=4, 
                top_p=0.95,
                frequency_penalty=0, 
                presence_penalty=0,
                logprobs=5
            )
        GPT3_CACHE[prompt] =response['choices'][0]
        predicted_text = response['choices'][0]['text'] # type: str
        predicted_logprob = response['choices'][0]['logprobs']['top_logprobs'] # type: List[Dict[str, float]]
        return predicted_text, predicted_logprob
