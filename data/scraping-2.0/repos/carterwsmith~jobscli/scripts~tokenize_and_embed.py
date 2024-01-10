import json
from math import floor
import os 

from dotenv import load_dotenv
import openai
from openai.embeddings_utils import get_embedding
import pandas as pd
import tiktoken

from .resume_utils import get_text

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
encoding = tiktoken.get_encoding(embedding_encoding)

def embed_jobs_from_path(filepath):
    # get the filename (remove rest of path, remove extension)
    filename = filepath.split("/")[-1].split(".")[0]

    # load dataset
    df = pd.read_json(filepath)

    df["combined"] = (
        "Title: " + df.title.str.strip() + "; Company: " + df.company.str.strip() + "; Info: " + df["info"].str.strip()
    )

    # append the location to df["combined"] in the rows where it exists
    df.loc[df.location.notnull(), "combined"] += "; Location: " + df.loc[df.location.notnull(), "location"].str.strip()

    # omit postings that are too long to embed
    df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens]
    # print number of valid postings and total number of tokens
    print(len(df), df.n_tokens.sum())

    df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))

    return df

def embed_jobs_from_dict(job_dict):
    # load dataset
    df = pd.DataFrame.from_dict(job_dict)
    #print(df)

    df["combined"] = (
        "Title: " + df.title.str.strip() + "; Company: " + df.company.str.strip() + "; Info: " + df["info"].str.strip()
    )

    # append the location to df["combined"] in the rows where it exists
    df.loc[df.location.notnull(), "combined"] += "; Location: " + df.loc[df.location.notnull(), "location"].str.strip()
    
    # omit postings that are too long to embed
    df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens]
    # print number of valid postings and total number of tokens
    # print(len(df), df.n_tokens.sum())

    df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))

    return df

def _cost_from_num_tokens(n):
    # $0.0001 per 1000 tokens
    return n / 1000 * 0.0001

def embed_resume(filepath):
    # get the filename (remove rest of path, remove extension)
    filename = filepath.split("/")[-1]

    text = get_text(filepath)

    # check resume is not too long
    n_tokens = len(encoding.encode(text))
    while n_tokens > max_tokens:
        print(f"Shaving tokens... [{n_tokens} > {max_tokens}]")
        # remove the last 1/4 of text
        text = text[:floor(len(text) * 3 / 4)]
        n_tokens = len(encoding.encode(text))
    resume_embed_cost = _cost_from_num_tokens(n_tokens)
    
    embedding = get_embedding(text, engine=embedding_model)

    return embedding, resume_embed_cost