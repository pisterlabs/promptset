from __future__ import annotations
from math import nan
import itertools
import os
import dotenv
import json
import pathlib
import openai
from typing import Union, Dict

Usage = Dict[str, Union[int, float]]   # TODO: could have used Counter class
default_usage_file = pathlib.Path("usage_openai.json")

# prices match https://openai.com/pricing as of November 2023.
pricing = {
        'gpt-3.5-turbo-1106':        { 'input': 0.0010, 'output': 0.0020, },
        'gpt-3.5-turbo-instruct':    { 'input': 0.0015, 'output': 0.0020, },
        'text-embedding-ada-002-v2': { 'input': 0.0001, 'output': 0.0001, },  # for embedding, output should never be used
        'gpt-4-1106-preview':        { 'input': 0.01,   'output': 0.03, },
        'gpt-4':                     { 'input': 0.03,   'output': 0.06, },
        'gpt-4-32k':                 { 'input': 0.06,   'output': 0.12, },
    }
default_model = "gpt-3.5-turbo-1106"
default_eval_model = "gpt-4-1106-preview"   # use a better model for evaluation

def track_usage(client: openai.OpenAI, path: pathlib.Path = default_usage_file) -> openai.OpenAI:
    """
    This method modifies (and returns) `client` so that its API calls
    will log token counts to `path`. If the file does not exist it
    will be created after the first API call. If the file exists the new 
    counts will be added to it.  
    
    The `read_usage()` function gets a Usage object from the file, e.g.:
    {
        "completion_tokens": 20,
        "prompt_tokens": 30,
        "total_tokens": 50,
        "cost": 0.00002
    }
    
    >>> client = openai.OpenAI()
    >>> track_usage(client, "example_usage_file.json")
    >>> type(client)
    <class 'openai.OpenAI'>
    
    """
    old_completion = client.chat.completions.create
    def tracked_completion(*args, **kwargs):
        response = old_completion(*args, **kwargs)
        old: Usage = read_usage(path)
        new: Usage = get_usage(response)
        _write_usage(_merge_usage(old, new), path)
        return response

    old_embed = client.embeddings.create
    def tracked_embed(*args, **kwargs):
        response = old_embed(*args, **kwargs)
        old: Usage = read_usage(path)
        new: Usage = get_usage(response)
        _write_usage(_merge_usage(old, new), path)
        return response

    client.chat.completions.create = tracked_completion    # type:ignore
    client.embeddings.create = tracked_embed   # type:ignore
    return client

def get_usage(response) -> Usage:
    """Extract usage info from an OpenAI response."""
    usage: Usage = vars(response.usage).copy()
    try:
        costs = pricing[response.model]
    except KeyError:
        raise ValueError(f"Don't know prices for model {response.model}")

    cost = (  usage.get('prompt_tokens',0)     * costs['input']
            + usage.get('completion_tokens',0) * costs['output']) / 1000
    usage['cost'] = cost

    return usage    

def read_usage(path: pathlib.Path = default_usage_file) -> Usage:
    """Retrieve total usage logged in a file."""
    if os.path.exists(path):
        with open(path, "rt") as f:
            return json.load(f)
    else:
        return {}

def _write_usage(u: Usage, path: pathlib.Path):
    with open(path, "wt") as f:
        json.dump(u, f, indent=4)

def _merge_usage(u1: Usage, u2: Usage) -> Usage:
    return {k: u1.get(k, 0) + u2.get(k, 0) for k in itertools.chain(u1,u2)}
    
# Make a tracked client object that everyone can use, as a convenience.
dotenv.load_dotenv()                           # set environment variable OPENAI_API_KEY from .env
default_client = track_usage(openai.OpenAI())  # create a client and modify it so that it will store its usage in a local file 
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
