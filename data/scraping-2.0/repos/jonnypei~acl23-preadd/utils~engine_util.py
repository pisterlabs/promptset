import math
from typing import Dict, Optional, Union, Sequence

from scipy.special import expm1
import openai
from transformers import AutoTokenizer
import torch
import requests


server_tokenizer = None

def get_tokenizer(model_string):
    global server_tokenizer
    if server_tokenizer is None:
        server_tokenizer = AutoTokenizer.from_pretrained(model_string)
    return server_tokenizer


def get_next_logprobs(prompt, model_string, cache_id=None, include_indices=[]):
    # prompt should be a list of tokens
    assert type(prompt) == list
    if len(prompt) > 0:
        assert type(prompt[0]) == int
    return server_next_logprobs(prompt, model_string, cache_id=cache_id)


def server_next_logprobs(prompt, model_string, cache_id=None, url='http://localhost:9741/logits'):
    # prompt is just a list of ints, just doing 1 at a time for now
    data = {'prompt': [prompt], 'cache_id': cache_id}
    r = requests.post(url, json=data)
    response = r.json()
    return {'logits': response['logits'][0],
            'cache_id': response['cache_id']}