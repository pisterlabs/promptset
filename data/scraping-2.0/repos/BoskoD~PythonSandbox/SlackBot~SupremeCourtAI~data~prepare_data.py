from config import KEY_CONFIG
import openai
from google.colab import drive
import pandas as pd
import tiktoken 
from openai.embeddings_utils import get_embedding

# tiktoken needs python 3.8
# https://github.com/acheong08/ChatGPT/issues/573

# openai api key - SET YOUR KEY HERE
openai.api_key = KEY_CONFIG.get("OPENAI_KEY")

# access google drive
drive.mount('/content/drive')

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# load & inspect dataset - here we have a folder called datasets in my Google drive
input_datapath = "/content/drive/MyDrive/datasets/justice.csv"
df = pd.read_csv(input_datapath, index_col=0)
# select specific columns from the data frame
df = df[["name", "term", "first_party", "second_party", "facts", "first_party_winner", "disposition"]]
# drop any rows with missing values
df = df.dropna()
# create a new column with combined values
df["combined"] = (
    "Name: " + df.name.str.strip() + "; Term: " + df.term + "; First Party: " + df.first_party.str.strip() + "; Second Party: " + df.second_party.str.strip() + "; First Party Winner: " + str(df.first_party_winner) + "; Disposition: " + df.disposition.str.strip() + "; Facts: " + df.facts.str.strip()
)
# print first two rows to see what the dataset looked like
df.head(2)

# get_encoding takes text as input and returns its corresponding token encoding
encoding = tiktoken.get_encoding(embedding_encoding)

# create a new column n_tokens
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))

# omit reviews that are too long to embed 
df = df[df.n_tokens <= max_tokens]

# print length of dataframe to make sure it isn't too long
len(df)

# condense the file
df = df.head(2000)

# We're saving a new file with the embedding arrays - this may take a few minutes - 2000 rows cost about $0.12
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))
df.to_csv("/content/drive/MyDrive/datasets//justice_supreme_court_cases_new.csv")