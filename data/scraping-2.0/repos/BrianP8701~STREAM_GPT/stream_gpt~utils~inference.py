import openai
import os
import json
import warnings
from stream_gpt.constants import prompts, function_schemas
from stream_gpt.constants.keys import OPENAI_API_KEY
from stream_gpt.utils import helpers

openai.api_key = OPENAI_API_KEY

def chat_with_gpt3_turbo(messages, temperature=0.0):
    if type(messages) == str: # In case someone accidentally passes in a string instead of a list of messages
        warnings.warn("chat_with_gpt3_turbo() expects a list of messages, not a string.")
        messages = [{"role": "user", "content": messages}]
    for message in messages:
        message["content"] = message["content"].encode('latin-1', errors='ignore').decode('latin-1')
    completion = openai.ChatCompletion.create(model='gpt-3.5-turbo-16k',messages=messages,temperature=temperature)
    return completion

def function_call_with_gpt3_turbo(messages, functions, function_call='auto', temperature=0.0):
    if type(messages) == str: # In case someone accidentally passes in a string instead of a list of messages
        warnings.warn("chat_with_gpt3_turbo() expects a list of messages, not a string.")
        messages = [{"role": "user", "content": messages}]
    for message in messages:
        message["content"] = message["content"].encode('latin-1', errors='ignore').decode('latin-1')
    completion = openai.ChatCompletion.create(model='gpt-3.5-turbo-16k',messages=messages,temperature=temperature,functions=functions, function_call=function_call)
    return completion

def chat_with_gpt3_instruct(prompt, temperature=0.0):
    if type(prompt) == list: # In case someone accidentally passes in a list of messages instead of a prompt
        warnings.warn("chat_with_gpt3_instruct() expects a prompt, not a list of messages.")
        prompt = '\n'.join(f'{message["role"]}: {message["content"]}' for message in prompt)
    response = openai.Completion.create(model="gpt-3.5-turbo-instruct",prompt=prompt, temperature=temperature)
    return response

def summarize(user_prompt, text, model="gpt-3.5-turbo-instruct"):
    if model == "gpt-3.5-turbo-instruct":
        prompt = f'{prompts.KEYWORD_SUMMARIZATION} {user_prompt}\n{text}'
        response = chat_with_gpt3_instruct(prompt).choices[0].text
    if model == "gpt-3.5-turbo-16k":
        messages=[
            {"role": "system", "content": f'{prompts.SUMMARIZATION} {user_prompt}'},
            {"role": "user", "content": text}
        ]
        response = chat_with_gpt3_turbo(messages).choices[0]['content']
    return response

def rank_categories(user_prompt, categories, model='gpt-3.5-turbo-16k'):
    '''
        Compare and rank a list of categories based on how well they match the user's prompt.
        Must use a model that supports function calling.
        
        Args:
        - user_prompt (string): Prompt from user
        - categories  (list): List of categories to compare and rank
        - model       (string): Model to use for inference
        
        Returns:
        - ranked_categories (list): List of categories ranked by relevance
    '''
    messages = [{"role": "user", "content": user_prompt},
                {"role": "user", "content": helpers.concatenate_with_indices(categories)}]
    response = function_call_with_gpt3_turbo(messages, function_schemas.RANK_CATEGORIES, function_call={'name':'rank_categories'}).choices[0]['message']['function_call']['arguments']
    return(json.loads(response))
    
def choose_best_scraped_text(samples):
    '''
        When using pdf scrapers, sometimes noise can happen. Here we ask ChatGPT 
        to choose the best sample from a list of samples.
        
        Args:
        - samples (list): List of sample from each scraper. Each sample is a string.
    '''
    user_prompt = ''
    index = 1
    for sample in samples:
        user_prompt += f'{index}: {sample}\n'
    messages = [{"role": "user", "content": user_prompt}]
    response = function_call_with_gpt3_turbo(messages, function_schemas.CHOOSE_BEST_SAMPLE, function_call={'name':'choose_best_sample'}).choices[0]['message']['function_call']['arguments']
    return(json.loads(response)['best_sample'])


