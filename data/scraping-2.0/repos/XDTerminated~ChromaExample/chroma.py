from chromadb.utils import embedding_functions
import io
import os
import uuid
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
import streamlit as st


import sys
from langchain.document_loaders import TextLoader
from langchain.document_loaders import CSVLoader

from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes import VectorstoreIndexCreator

import chromadb

# Sets up environ variables for openai
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = "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"

# Creates a new embedding function (OpenAI Embedding Function)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-ada-002",
    api_key="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",
)

# Gets the chroma client
chroma_client = chromadb.HttpClient(host="localhost", port=8000)


# Deletes a collection in the databse
def delete_collection(collection):
    chroma_client.delete_collection(name=collection)


# Creates a collection in the databse
def create_collection(collection):
    chroma_client.create_collection(name=collection, embedding_function=openai_ef)


# Gets a collection in the databse
def get_collection(collection_name):
    return chroma_client.get_collection(name=collection_name)


def upload_file(file_name, collection):
    text = ""
    with io.open(file_name, "r") as f:
        text = f.read()

    collection.add(documents=[text], metadatas=[{"source": file_name}], ids=[uid()])


# Does a similarity search between the prompt and the data in the collection
def query_chroma(collection, prompt):
    query = collection.query(
        query_texts=[
            prompt,
        ],
        n_results=4,
    )

    return query


def get_all_collections():
    return chroma_client.list_collections()


# Generates a random ID
def uid():
    return str(uuid.uuid4())


# Uploads a file to the collection


results = query_chroma(get_collection("test"), "What is the most common rating?")

# a = results["documents"][0]


# with open("temp.csv", "w") as f:
#     f.write(a[0] + "\n")

# a = pd.read_csv("temp.csv")

# agent = create_pandas_dataframe_agent(
#     OpenAI(temperature=0, max_tokens=100), a, verbose=True
# )

# print(agent.run("What is the most common opening?"))

query = sys.argv[1]
print(query)
loader = TextLoader("temp.csv")
index = VectorstoreIndexCreator().from_loaders([loader])
print(index.query(query))
