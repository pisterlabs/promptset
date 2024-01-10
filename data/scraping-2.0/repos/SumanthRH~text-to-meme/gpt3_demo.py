"""
Creates a PrimeGPT class to interface with the OpenAI API.
Helper functions borrowed from https://github.com/shreyashankar/gpt3-sandbox
"""
import openai
import pickle
from utils.gpt import Example, GPT
import numpy as np

def clean_and_unify_caption(caption):
    return caption[0].strip()+'<sep>'+caption[1].strip()

def read_data(file_path):
    # load data
    with open(file_path, 'rb') as f:
        gpt3_data = pickle.load(f)
    return gpt3_data

def set_openai_key(key):
    """Sets OpenAI key."""
    openai.api_key = key

class PrimeGPT(object):
    def __init__(self, api_key, gpt3_data_path, gpt3_engine, temperature, max_tokens):
        set_openai_key(api_key)
        self.gpt = GPT(engine=gpt3_engine, temperature=temperature, max_tokens=max_tokens)
        self.gpt3_data = read_data(gpt3_data_path)
        
    def clear_gpt_examples(self):
        self.gpt.examples = {}
    
    def prime_gpt_from_uuid(self, uuid):
        self.clear_gpt_examples()
        tuples = self.gpt3_data[uuid]
        for (user_prompt, meme_caption) in tuples:
            meme_caption = clean_and_unify_caption(meme_caption)
            self.gpt.add_example(Example(user_prompt, meme_caption))
    
    def get_response(self, uuid, prompt):
        self.prime_gpt_from_uuid(uuid)
        return self.gpt.submit_request(prompt)