from datasets import load_dataset
import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding
import openai
import csv

# openai.api_key = ""

"""Helper methods to get texts and embedding"""
def output_dataset_to_csv(path="./recsim_ng/data/ag_news_train.csv"):
    file = open(path, 'w')
    writer = csv.writer(file)
    # header line
    writer.writerow(["text", "label"])
    for item in load_dataset("ag_news", split="train"):
        data = [item["text"], item["label"]]
        writer.writerow(data)

def output_all_embedding_to_csv(num, path="./recsim_ng/data/ag_news_train.csv"):
    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
    max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
    df = pd.read_csv(path)
    # num = -1 means encodes all data to embeddings
    if num != -1:
        df = df[["text", "label"]].tail(num*2) # first cut to first 2k entries, assuming less than half will be filtered out
    else:
        df = df[["text", "label"]] 

    encoding = tiktoken.get_encoding(embedding_encoding)

    # omit text that are too long to embed
    df["n_tokens"] = df.text.apply(lambda x: len(encoding.encode(x)))
    if num != -1:
        df = df[df.n_tokens <= max_tokens].tail(num)
    else:
        df = df[df.n_tokens <= max_tokens]

    df["embedding"] = df.text.apply(lambda x: get_embedding(x, engine=embedding_model))
    # Notice: the embedding has 1536 dimensions
    df.to_csv("./recsim_ng/data/ag_news_train_embeddings.csv")

