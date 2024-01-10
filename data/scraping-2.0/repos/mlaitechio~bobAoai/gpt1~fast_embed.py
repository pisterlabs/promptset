
from ast import literal_eval
import concurrent
import openai
import os
import numpy as np
from numpy import array, average
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from tqdm import tqdm
from typing import List, Iterator
# import wget
# import Redis
# Redis imports
# from redis import Redis as r
# from redis.commands.search.query import Query
# from redis.commands.search.field import (
#     TextField,
#     VectorField,
#     NumericField
# )

# from redis.commands.search.indexDefinition import (
#     IndexDefinition,
#     IndexType
# )
from dotenv import load_dotenv
# Langchain imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
# from API_data import merge_csv
import sys
import django
import subprocess
# sys.path.append("/".join(os.getcwd().replace("\\", "/").split("/")[0:-1]))
# os.environ['DJANGO_SETTINGS_MODULE'] = 'testopenAI.settings'
# django.setup()

load_dotenv()
dir_path = ""
# dir_path ="abslamcchatgpt"
# dir_path = "/mnt/c/Users/Nexgits/Desktop/chatbot-web-socket-main"
# csv_file = ".csv"
# csv_path = os.path.abspath(os.path.)
# API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY") 
RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") 
# openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
TEXT_EMBEDDING_CHUNK_SIZE = 2500
openai.api_version = "2022-12-01"

tokenizer = tiktoken.get_encoding("cl100k_base")
## Chunking Logic

# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j
        
def get_unique_id_for_file_chunk(title, chunk_index):
    return str(title+"-!"+str(chunk_index))

def chunk_text(x,text_list):
    
    title = str(x['title'])
    file_body_string = str(x['content'])
        
    """Return a list of tuples (text_chunk, embedding) for a text."""
    token_chunks = list(chunks(file_body_string, TEXT_EMBEDDING_CHUNK_SIZE, tokenizer))
    text_chunks = [f'Fund: {title};\nFund Detail'+ tokenizer.decode(chunk) for chunk in token_chunks]
    
    #embeddings_response = openai.Embedding.create(input=text_chunks, model=EMBEDDINGS_MODEL)

    #embeddings = [embedding["embedding"] for embedding in embeddings_response['data']]
    #text_embeddings = list(zip(text_chunks, embeddings))

    # Get the vectors array of triples: file_chunk_id, embedding, metadata for each embedding
    # Metadata is a dict with keys: filename, file_chunk_index
    
    for i, text_chunk in enumerate(text_chunks):
        id = get_unique_id_for_file_chunk(title, i)
        text_list.append(({'id': id
                         , 'metadata': {"Fund": str(title)
                                      , "Fund Detail": str(text_chunk)
                                      , "file_chunk_index": i}}))
        
## Batch Embedding Logic
EMBEDDINGS_MODEL = "bob-text-embedding-ada-002"
# Simple function to take in a list of text objects and return them as a list of embeddings
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(10))
def get_embeddings(input: List):
    
    response = openai.Embedding.create(
        engine='bob-text-embedding-ada-002',
        input=input )["data"]
    return [data["embedding"] for data in response]
# def get_embeddings(input: List):
    
#     response = openai.Embedding.create(
#         input=input,
#         engine=EMBEDDINGS_MODEL,
#     )["data"]
#     return [data["embedding"] for data in response]

def batchify(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]

# Function for batching and parallel processing the embeddings
def embed_corpus(
    corpus: List[str],
    batch_size=64,
    num_workers=8,
    max_context_len=8191,
):

    # Encode the corpus, truncating to max_context_len
    encoding = tiktoken.get_encoding("cl100k_base")
    encoded_corpus = [
        encoded_article[:max_context_len] for encoded_article in encoding.encode_batch(corpus)
    ]

    # Calculate corpus statistics: the number of inputs, the total number of tokens, and the estimated cost to embed
    num_tokens = sum(len(article) for article in encoded_corpus)
    cost_to_embed_tokens = num_tokens / 1_000 * 0.0004
    print(
        f"num_articles={len(encoded_corpus)}, num_tokens={num_tokens}, est_embedding_cost={cost_to_embed_tokens:.2f} USD"
    )

    # Embed the corpus
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        
        futures = [
            executor.submit(get_embeddings, text_batch)
            for text_batch in batchify(encoded_corpus, batch_size)
        ]

        with tqdm(total=len(encoded_corpus)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(batch_size)

        embeddings = []
        for future in futures:
            data = future.result()
            embeddings.extend(data)

        return embeddings
    
    



# Process each PDF file and prepare for embedding
def embeded_csv():
    text_list = []
    input_filename = "Bob_bank_token.csv"
    df = pd.read_csv(input_filename,encoding="utf-8")
    x = df.apply(lambda x: chunk_text(x, text_list),axis = 1)
    df3 = pd.DataFrame.from_records(text_list)
    
    df3["token"] = None
    for idx, r in df3.iterrows():
    #     print(len(r.content))
    #     df["token"] = df[len(r.content)]
        mt = str(r.metadata)
        df3.loc[idx,'token'] = len(mt)
    df3.rename(columns = {'metadata':'content'}, inplace = True)   
    df3.rename(columns = {'id':'title'}, inplace = True) 
    chunk_filename = os.path.abspath(os.path.join(dir_path,"Bob_chunks.csv"))
    df3.to_csv(chunk_filename,encoding="utf-8")
    # print(text_list[25])
    embeddings = embed_corpus([text["metadata"]['Fund Detail'] for text in text_list])

    # print(len(embeddings))
    embedding = {}
    for i,x in enumerate(embeddings):
        embedding[i] = x
        
    df2 = pd.DataFrame.from_dict(embedding, orient='index')
    output_filename = os.path.abspath(os.path.join(dir_path,"Bob_embedded.csv"))
    df2.index.names = ['title']
    df2.columns = [str(i) for i in range(df2.shape[1])]
    # columns = [len(values) for i in ranfe(df.shape[1])]
    # print(columns)
    # Replace `filename.csv` with the desired filename/path
    df2.to_csv(output_filename)
    
    # subprocess.call(['sudo', 'systemctl', 'reboot'])
    return chunk_filename , output_filename
    
# filename = api_csv()
embeded_csv()
