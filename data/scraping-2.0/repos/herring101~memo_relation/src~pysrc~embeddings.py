import h5py
import numpy as np
import sys
import openai
import os
from dotenv import load_dotenv

load_dotenv()


def get_embedding(text, model="text-embedding-ada-002"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def calculate_embeddings(file):
    # Load the data(md)
    with open(file, "r") as f:
        data = f.readlines()

    # get topic
    topic = data[0].strip()[2:]

    # calculate embeddings
    embeddings = get_embedding(" ".join(data))

    return topic, embeddings


file = sys.argv[1]

topic, embeddings = calculate_embeddings(file)

with h5py.File("embeddings.hdf5", "a") as f:
    group = f.create_group(file)
    group.create_dataset("topic", data=topic)
    group.create_dataset("embeddings", data=embeddings)
