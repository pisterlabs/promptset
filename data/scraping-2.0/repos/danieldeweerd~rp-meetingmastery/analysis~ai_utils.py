import json
import os

import numpy as np
import openai
import requests
import tiktoken

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")


def get_number_of_tokens(string):
    string = string.replace("\n", " ")
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(string))


def get_embedding(string, as_numpy=True, use_ada=False):
    if use_ada:
        return get_ada_embedding(string)
    else:
        return get_minilm_embedding(string, as_numpy=as_numpy)


def get_minilm_embedding(string, as_numpy=True):
    string = string.replace("\n", " ")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding = model.encode([string])[0]

    return np.array(embedding) if as_numpy else embedding


def get_minilm_embedding_batch(strings):
    for i, string in enumerate(strings):
        strings[i] = string.replace("\n", " ")

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(strings)


def get_ada_embedding(string, use_numpy=True):
    string = string.replace("\n", " ")
    embedding = openai.Embedding.create(input=[string], model="text-embedding-ada-002")['data'][0]['embedding']
    return np.array(embedding) if use_numpy else embedding


def get_ada_embedding_batch(strings):
    for i, string in enumerate(strings):
        strings[i] = string.replace("\n", " ")

    embeddings = openai.Embedding.create(input=strings, model="text-embedding-ada-002")
    return _ada_embeddings_to_matrix(embeddings)


def _ada_embeddings_to_matrix(embeddings, m=1536):
    embeddings = embeddings["data"]
    n = len(embeddings)
    matrix = np.zeros((n, m))
    for i, embedding in enumerate(embeddings):
        matrix[i] = embedding["embedding"]

    return matrix


def classify_expression(thesis, expression):
    template = """
    You are given a thesis and an expression. Output 'D' if the expression agrees with the thesis. Output 'D' if the expression disagrees with the thesis. 
    Output 'O' if the expression neither agrees nor disagrees with the thesis. Limit your output to the single character.

    Thesis: {}
    Expression: {}
    """

    prompt = template.format(thesis, expression)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}])

    return response['choices'][0]['message']['content']
