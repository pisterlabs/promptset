# import required modules
import openai
import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding

# Create a requirements.txt file
with open('requirements.txt', 'w') as f:
    f.write("""openai
tiktoken
pandas
transformers
plotly
matplotlib
sklearn
torch
torchvision
scipy
""")

# Remember to run pip install

# Set openai api key
openai.api_key ='SETYOURKEYHERE'

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
max_tokens = 8000

# Note: set the path to your datafile
input_datapath = "YOUR_PATH_HERE/datasets/justice.csv"
df = pd.read_csv(input_datapath, index_col=0)

# Continue processing data
df = df[["name", "term", "first_party", "second_party", "facts", "first_party_winner", "disposition"]]
df = df.dropna()
df["combined"] = (
    "Name: " + df.name.str.strip() + "; Term: " + df.term + "; First Party: " + df.first_party.str.strip() + "; Second Party: " + df.second_party.str.strip() + "; First Party Winner: " + str(df.first_party_winner) + "; Disposition: " + df.disposition.str.strip() + "; Facts: " + df.facts.str.strip()
)

# get_encoding takes text as input and returns its corresponding token encoding
encoding = tiktoken.get_encoding(embedding_encoding)
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens]

# condense the file
df = df.head(2000)

# Get embeddings
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))
# Save the DataFrame df to a CSV file at the specified path.
df.to_csv("YOUR_PATH_HERE/data/justice_supreme_court_cases_new.csv")

