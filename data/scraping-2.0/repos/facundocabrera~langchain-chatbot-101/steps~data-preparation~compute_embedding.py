import time
import pandas as pd
import os
import openai

from pathlib import Path
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv

load_dotenv()  # This will load the variables from .env file

openai.api_key = os.environ["OPENAI_API_KEY"]

# i/o files
input_datapath = Path(__file__) / ".." / ".." / ".." / "data" / "movies_tokens.csv"

# this time we prefer json for the next step
output_datapath = Path(__file__) / ".." / ".." / ".." / "data" / "movies_embedding.json"

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002

# check https://platform.openai.com/account/rate-limits
max_tokens = 1000000 # the maximum text-embedding-ada-002 tokens per minute
max_requests = 3000  # the maximum number of requests per minute

print( "input path:", input_datapath.resolve() )
print( "output path:", output_datapath.resolve() )

def get_embedding_with_rate_limit(text):
    while True:
        try:
            embedding = get_embedding(text, engine=embedding_model)
            break
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded. Waiting for {e.retry_after} seconds...")
            time.sleep(e.retry_after)
    return embedding

# read the data from previous step
df = pd.read_csv(input_datapath.resolve(), index_col=0)

# add a new column with list type
df["embedding"] = [None] * len(df.index)

# keep the first 10 items for testing
# df = df.head(10)

# count how many tokens each row has to rate limit the execution
n_tokens_acc = 0

# iterate over the rows to process the embedding column
for i, row in df.iterrows():
    print(f"Processing ", row["n_tokens"], " tokens")

    df.at[i, "embedding"] = get_embedding_with_rate_limit(row["combined"])
    
    n_tokens_acc += row["n_tokens"]

    if n_tokens_acc > max_tokens:
        print(f"Max tokens per minute exceeded. Waiting for 60 seconds before continue with the next batch...")
        time.sleep(60)
        n_tokens_acc = 0

# save to json for simplicity (remove indexes from the output).
df.to_json(output_datapath.resolve(), orient="records")
