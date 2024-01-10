# imports
import os

import openai
import pandas as pd
import tiktoken

from openai.embeddings_utils import get_embedding

# setting api key
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv("OPENAI_API_KEY")

# embedding model parameters
embedding_model = "text-embedding-ada-002"

# load & inspect dataset
input_datapath = "Jira_data_1k.csv"  # to save space, we provide a pre-filtered dataset
df = pd.read_csv(input_datapath, index_col=0)
df = df[["jiraComments", "rca"]]
df = df.dropna()


# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage

# This may take a few minutes
df["embedding"] = df.jiraComments.apply(lambda x: get_embedding(x, engine=embedding_model))
df.to_csv("Jira_data_with_embeddings_1k.csv")

