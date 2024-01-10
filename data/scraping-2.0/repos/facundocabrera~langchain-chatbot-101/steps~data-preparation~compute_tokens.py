import pandas as pd
import tiktoken
import os
import openai

from pathlib import Path
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv

load_dotenv()  # This will load the variables from .env file

openai.api_key = os.environ["OPENAI_API_KEY"]

# i/o files
input_datapath = Path(__file__) / ".." / ".." / ".." / "data" / "movies_combined.csv"
output_datapath = Path(__file__) / ".." / ".." / ".." / "data" / "movies_tokens.csv"

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

print( "input path:", input_datapath.resolve() )
print( "output path:", output_datapath.resolve() )

df = pd.read_csv(input_datapath.resolve(), index_col=0)

encoding = tiktoken.get_encoding(embedding_encoding)

df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))

# omit movies that are too long to embed
df = df[df.n_tokens <= max_tokens]

df.to_csv(output_datapath.resolve())