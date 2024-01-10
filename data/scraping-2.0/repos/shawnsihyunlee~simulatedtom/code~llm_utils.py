# Author: Shawn Lee
# Date: Aug 2023
# Description: Wrappers for easy querying of LLMs.

CACHE_DIR = '/scratch/weights/llama2'
import os
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
import random
import csv
import tqdm
import argparse
import torch
import itertools
import wandb
from transformers import GenerationConfig, pipeline
import openai
import time
from typing import *

# Please provide the api key in api_key.txt!
with open("api_key.txt", "r") as f:
    API_KEY = f.readline().strip()
openai.api_key = API_KEY


class LLM:
    
    """LLM wrapper class.
        Meant for use with local Llama2-based HuggingFace models.
    """
    
    def __init__(
                self,
                model_name,                             # Model name to load
                load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
                device_map="auto",                      # Device mapping (GPU by default)
                max_new_tokens=1024,                    # Maximum number of new tokens to generate
                temperature=0.3,                        # Temperature setting for generation
                repetition_penalty=1.2,                 # Penalty for repeating tokens
                top_p=0.95,                             # Top-p for nucleus sampling
                top_k=50,                               # Top-k tokens considered for generation
                do_sample=False,                        # Whether to use sampling in generation
                cache_dir=CACHE_DIR,    # Directory to cache the weights
                gpu=0,                                  # GPU to use
                verbose=False,                          # Verbosity flag
            ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)
        
        if not load_in_8bit:
            device_kwargs = {"device_map":gpu}
        else:
            device_kwargs = {"device_map":gpu, "load_in_8bit": True}
        
        # Creating the generator using the specified model and device settings
        self.generator = pipeline(model=model_name, model_kwargs=device_kwargs, max_new_tokens=max_new_tokens)
    
    
    def getOutput(self, prompt):
        # Internal function to format the prompt
        def formatPrompt(prompt):
            # Template for chat model
            promptFormat="""\
<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST]"""
            # Predefined system prompt
            system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
            formattedPrompt = promptFormat.format(system_prompt=system_prompt, user_message=prompt)
            
            return formattedPrompt
        
        if self.verbose:
            print("### PROMPT ###")
            print(prompt)
        
        # For non-instruction tuned models, we don't follow the chat prompt.
        if "chat" in self.model_name:
            prompt = formatPrompt(prompt)
        eosToken = "[/INST]"
        
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        
        # self.generator returns the entire generation, including the original prompt; hence we chop off.
        if "chat" in self.model_name:
            response = self.generator(prompt, generation_config=generation_config)[0]['generated_text'].split("[/INST]")[1]
        else:
            response = self.generator(prompt, generation_config=generation_config)[0]['generated_text'][len(prompt):]
        
        if self.verbose:
            print("### RESPONSE ###")
            print(response)
        
        return response
    


class ChatGPT:
    """ChatGPT wrapper.
    """
    def __init__(self, model_name, api_key=None, temperature=0.0, verbose=False):
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        if api_key != None:
            openai.api_key = api_key # Setting OpenAI API key if provided
        
    def getOutput(self, prompt:str, max_retries=30) -> str:
        """Gets output from OpenAI ChatGPT API.

        Args:
            prompt (str): Prompt
            max_retries (int, optional): Max number of retries for when API call fails. Defaults to 30.

        Returns:
            str: ChatGPT response.
        """
        
        if self.verbose:
            print("### PROMPT ###")
            print(prompt)
        
        m = [
            {"role": "system", "content": "You are a helpful assistant."},
            {'role': 'user', 'content': prompt},
        ]

        for i in range(max_retries):
            try:
                res = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=m,
                    temperature = self.temperature
                )

                output = res.choices[0].message.content
                
                if self.verbose:
                    print("### RESPONSE ###")
                    print(output)
                
                return output
            except openai.error.OpenAIError as e:
                # Exponential backoff
                if i == max_retries - 1:  # If this was the last attempt
                    raise  # re-throw the last exception
                else:
                    # Wait for a bit before retrying and increase the delay each time
                    sleep_time = (2 ** i) + random.random()  # Exponential backoff with full jitter
                    time.sleep(sleep_time)
