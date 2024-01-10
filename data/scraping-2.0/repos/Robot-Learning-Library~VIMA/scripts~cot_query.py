import os
import openai
import wandb
import numpy as np
from private import *

def load_prompt(folder=''):
    prompts = np.load(folder)
    return prompts

def query_in_batch(prompts, wandb_activate=False): 
    openai.api_key = f"{OPENAI_API_KEY}"
    if wandb_activate:
        run = wandb.init(project='GPT-3 in Python')
        prediction_table = wandb.Table(columns=["prompt", "completion"])

    results = []
    for gpt_prompt in prompts:
        response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=gpt_prompt,
        temperature=0.9,  # https://beta.openai.com/docs/quickstart/adjust-your-settings
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )

        print(gpt_prompt, response['choices'][0]['text'])

        if wandb_activate:
            prediction_table.add_data(gpt_prompt,response['choices'][0]['text'])
        
        results.append(response['choices'][0]['text'])

    if wandb_activate:
        wandb.log({'predictions': prediction_table})
        wandb.finish()

    return results


def query(gpt_prompt, wandb_activate=False, temperature=0.8): 
    openai.api_key = f"{OPENAI_API_KEY}"
    if wandb_activate:
        run = wandb.init(project='GPT-3 in Python')
        prediction_table = wandb.Table(columns=["prompt", "completion"])

    response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=gpt_prompt,
    temperature=temperature,  # https://beta.openai.com/docs/quickstart/adjust-your-settings
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )

    print('Query: ', gpt_prompt, 'Response: ', response['choices'][0]['text'])

    if wandb_activate:
        prediction_table.add_data(gpt_prompt,response['choices'][0]['text'])
    
    if wandb_activate:
        wandb.log({'predictions': prediction_table})
        wandb.finish()

    return response['choices'][0]['text']

