import chromadb
import pandas as pd
import os
from chromadb.config import Settings
import markdown2
from bs4 import BeautifulSoup
from transformers import GPT2TokenizerFast
import numpy as np
import openai
from openai import OpenAI
import pickle
import numpy as np
from nltk.tokenize import sent_tokenize
import glob
from data_frame_vectorizer import vectorize_data
from chromadb.utils import embedding_functions

# Define embedding function
ef = embedding_functions.DefaultEmbeddingFunction()


# Add data straight from a dataframe
def add_data_pandas(
    df: pd.DataFrame,
    filename: str,
    client: chromadb.PersistentClient,
    collection_name: str,
):
    """
    Takes in a pandas dataframe and adds it to the chroma db

    Keyword Arguments:
    df -- A Pandas DataFrame that has previously been vectorized and is ready to be added
    filename -- The file that the data came from. This is used for creating indices
    client -- A chromadb client(can be either Normal or Persistent) for the data to be added to
    collection_name -- Location within the chromadb for the data to be added to
    """

    # Verify there is data to add
    if not list(df.index.values):
        print(
            f"*****************\n{filename} is empty, please check to verify data\n*****************"
        )
        return
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=ef
    )

    meta = []
    for i in df["heading"]:
        meta.append({filename: i})

    collection.add(
        documents=df["content"].tolist(),
        metadatas=meta,
        ids=[filename + str(i) for i in list(df.index.values)],
    )
    print(f"{filename} has been added to the Database!")
    return


chroma_client = chromadb.PersistentClient(path="./chroma_save_states")

vector_dataframes, filenames = vectorize_data("md_files", "vectorized_dataframes")
for i in range(len(vector_dataframes)):
    add_data_pandas(vector_dataframes[i], filenames[i], chroma_client, "testing")

collection = chroma_client.get_collection("testing")
collection.query(query_texts=["What is Abaca?"], n_results=5)

print(chroma_client)

chroma_client.hearbeat()
