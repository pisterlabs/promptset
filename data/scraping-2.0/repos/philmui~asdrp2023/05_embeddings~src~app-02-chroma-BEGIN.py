#
# Using Embeddings with Chroma and LlamaIndex
#
# Leaderboard for MTEB:
# https://huggingface.co/spaces/mteb/leaderboard
#

from pathlib import Path
import sys
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.vector_stores import ChromaVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
import chromadb
from chromadb.config import Settings

DATA_DIR = "../data"
STORAGE_DIR = "chroma_db"
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = "intfloat/e5-large-v2"
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def get_index():
    index = None

    embed_model = None # need to create our own embedding here
    
    chroma_client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=STORAGE_DIR)
    )
    try: 
        # get an existing chroma collection
        
        # instantiate a vector store for querying
        
        # create an index from the vector store
        pass
    except ValueError as ve:
        # did not get a valid chroma collection, let's create one
        
        # get news articles from our local files
        
        # create an index from the newly ingested docs
        
        # save the index
        pass
        
    return index
        
def get_response(index, query):
    query_engine = index.as_query_engine()
    return query_engine.query(query)

if __name__ == '__main__':

    index = get_index()
    while True:
        query = input("What question do you have? ('quit' to quit) ")
        if "quit" in query: break
        response = get_response(index, query)
        print(response)
