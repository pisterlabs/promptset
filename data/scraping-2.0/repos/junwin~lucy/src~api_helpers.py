import os
import json
import openai
import time
from openai.error import RateLimitError, InvalidRequestError
from src.container_config import container
from src.config_manager import ConfigManager
import logging

config = container.get(ConfigManager) 
credential_path = config.get('credential_path')
with open(credential_path + "/oaicred.json", "r") as config_file:
   config_data = json.load(config_file)

openai.api_key = config_data["openai_api_key"]




def ask_question(conversation, model="gpt-3.5-turbo", temperature=0, max_retries=3, retry_wait=1) -> str:
    logging.info(f'ask_question start: {model}')
    logging.info(f'before send: {conversation}')
    retries = 0
    while retries <= max_retries:
        try:
            myAns = openai.ChatCompletion.create(
                model=model,
                messages=conversation,
                temperature=temperature,
            )
            response = myAns.choices[0].message['content']
            response = response.encode('utf-8').decode('utf-8')
            logging.info(f'ask_question end: {model}')
            return response
        except RateLimitError as e:
            if retries == max_retries:
                raise e
            retries += 1
            print(f"RateLimitError encountered, retrying... (attempt {retries})")
            time.sleep(retry_wait)
        except InvalidRequestError as e:
            if retries == max_retries:
                raise e
            retries += 1
            print(f"InvalidRequestError encountered, retrying... (attempt {retries})")
            time.sleep(retry_wait)


def get_completion(prompt:str, temperature:int=0, model:str="gpt-3.5-turbo", max_retries:int=3, retry_wait:int=1):
    logging.info(f'get_completion start: {model}')
    messages = [{"role": "user", "content": prompt}]
    #messages = json.dumps(messages)
    logging.info(f'before send: {messages}')

    retries = 0
    while retries <= max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,  # this is the degree of randomness of the model's output
            )
            logging.info(f'get_completion end: {model}')
            return response.choices[0].message["content"]
            
        except RateLimitError as e:
            if retries == max_retries:
                raise e
            retries += 1
            print(f"RateLimitError encountered, retrying... (attempt {retries})")
            time.sleep(retry_wait)

def get_completionWithFunctions(messages ,functions:str, temperature:int=0, model:str="gpt-3.5-turbo-0613", max_retries:int=3, retry_wait:int=1):
    logging.info(f'get_completionWithFunctions start: model {model} functions {functions} messages {messages}')

    retries = 0
    while retries <= max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                functions=functions,
                function_call = "auto", 
                temperature=temperature,  # this is the degree of randomness of the model's output
            )
            logging.info(f'get_completion end: {model}')
            response_message = response["choices"][0]["message"]
            return response_message
            
        except RateLimitError as e:
            if retries == max_retries:
                raise e
            retries += 1
            print(f"RateLimitError encountered, retrying... (attempt {retries})")
            time.sleep(retry_wait)

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_retries=3, retry_wait=1):
    logging.info(f'get_completion_from_messages start: {model}')
    retries = 0
    while retries <= max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,  # this is the degree of randomness of the model's output
            )
            logging.info(f'get_completion_from_messages end: {model}')
            return response.choices[0].message["content"]
        except RateLimitError as e:
            if retries == max_retries:
                raise e
            retries += 1
            print(f"RateLimitError encountered, retrying... (attempt {retries})")
            time.sleep(retry_wait)
