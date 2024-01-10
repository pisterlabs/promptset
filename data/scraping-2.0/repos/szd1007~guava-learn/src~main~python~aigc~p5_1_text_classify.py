import pandas as pd
import tiktoken
import openai
import os
import backoff
from openai.embeddings_utils import get_embedding, get_embeddings

openai.api_key = os.environ.get("OPENAI_API_KEY")

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base" # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# import data/xx.txt as a pandas dataframe
df = pd.read_csv("/Users/zm/aigcData/toutiao-text-classfication-dataset/toutiao_cat_data.txt", sep='_!_', names=['id', 'code', 'category', 'title', 'keywords'])
df = df.fillna("")
df["combined"] = ("标题：" + df.title.str.strip() + "; 关键字：" + df.keywords.str.strip())

print("Lines of text before filtering: ", len(df))

batch_size = 1000

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embeddings_with_backoff(prompts, engine):
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        embeddings += get_embeddings(list_of_text=batch, engine=engine)
        return embeddings
# randomly sample 10k rows
df_all = df.sample(170, random_state=42)

prompts = df_all.combined.tolist()
prompts_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

embeddings = []
for batch in prompts_batches:
    batch_embeddings = get_embeddings_with_backoff(prompts=batch, engine=embedding_model)
    embeddings += batch_embeddings

df_all["embedding"] = embeddings
df_all.to_parquet("/Users/zm/aigcData/toutiao-text-classfication-dataset/toutiao_cat_data_all_with_embeddings_local.parquet")