import json
import openai
import os
openai.api_key = os.environ.get('OPENAI_API_KEY')

max_tokens = 8191
tokens_per_request = 40000

from config import Config

from ratelimit import limits, RateLimitException, sleep_and_retry
import time

ONE_MINUTE = 60
MAX_CALLS_PER_MINUTE = 3

current_request = 0

@sleep_and_retry
@limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
def get_embeddings(sentences):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=sentences,
    )
    embeddings = response['data']
    return embeddings


import tiktoken
import pandas as pd

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_strings(strings: list[str], encoding_name: str) -> int:
    """Returns the number of tokens in a list of text strings."""
    return sum(num_tokens_from_string(s, encoding_name) for s in strings)

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np

# Get openai embeddings for each node in the graph in batches
def get_embeddings_for_nodes(graph):
    global max_tokens, tokens_per_request

    embeddings = []
    batch = []
    tokens = 0

    for node in graph.nodes.values():
        text = node.text
        node_tokens = num_tokens_from_string(text, "cl100k_base")

        if node_tokens > max_tokens:
            text = text[:max_tokens]
        if node_tokens == 0:
            continue
        tokens += node_tokens

        if tokens > tokens_per_request:
            # make request
            batch = np.array(batch)
            response = get_embeddings(list(batch[:,2]))
            # add the embeddings to the dict           
            for i, res in enumerate(response):
                dict = {'number': batch[i,0], 'node_type': batch[i,1], 'embedding': res['embedding']}
                embeddings.append(dict)
            # reset tokens and batch
            tokens = 0
            batch = []
            # add the current node to the batch
            batch.append([node.number, node.node_type, text])
            tokens += node_tokens
        else:
            batch.append([node.number, node.node_type, text])

    if tokens > 0:
        # Batch the remaining nodes
        batch = np.array(batch)
        response = get_embeddings(list(batch[:,2]))
        # add the embeddings to the dict           
        for i, res in enumerate(response):
            dict = {'number': batch[i,0], 'node_type': batch[i,1], 'embedding': res['embedding']}
            embeddings.append(dict)

    f = open(f"data_{Config().repo_name}/embeddings.json", "w")
    dump = {'embeddings': embeddings}
    f.write(json.dumps(dump))
    f.close()

# Read the embeddings from the json file
def read_embeddings():
    f = open("embeddings.json", "r")
    embeddings = json.loads(f.read())
    f.close()
    embeddings = embeddings['embeddings']
    return embeddings