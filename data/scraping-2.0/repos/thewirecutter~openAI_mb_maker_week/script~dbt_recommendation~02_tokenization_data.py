# import csv data then perform some cleaning for the ai
# then create the embeddings, clearing way for the AI to learn

import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding

# set up model and tokenization
# embedding model parameters
# https://platform.openai.com/tokenizer
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# read in data
data_path = "/home/data"
cleaned_path = data_path + "/cleaned/"
processed_path = data_path + "/processed/"
df = pd.read_csv(cleaned_path + "dbt_nodes.csv")
df.dropna()
# add column name in front of every value : e.g database: wc_data_reporting
df = df.apply(lambda row: str(row.name) + ": " + row.astype(str), axis=0)
# create a combined field
df["combined"] = df.apply(lambda row: ";".join(str(val) for val in row), axis=1)
df["combined"][0]

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding(embedding_encoding)
df["n_tokens"] = df.combined.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.describe()

# create embeddings and save them for future use
df["embeddings"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))

df.to_csv(processed_path + "dbt_manifest_embeddings.csv")
