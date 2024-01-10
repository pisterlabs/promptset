#
# Using Embeddings with Chroma and LlamaIndex
#
# Chroma DB: https://www.trychroma.com/
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

    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )
    chroma_client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=STORAGE_DIR)
    )
    try:
        chroma_collection = chroma_client.get_collection("news")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context, embed_model=embed_model
        )
    except ValueError as ve:
        chroma_collection = chroma_client.create_collection("news")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        docs = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, embed_model=embed_model
        )
        chroma_client.persist()
        
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
