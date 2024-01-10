from datasets import load_dataset

data = load_dataset('jamescalam/youtube-transcriptions', split='train')

from lancedb.context import contextualize
df = (contextualize(data.to_pandas())
      .groupby("title").text_col("text")
      .window(20).stride(4)
      .to_df())
df.head(1)

import openai
import os

# Configuring the environment variable OPENAI_API_KEY
if "OPENAI_API_KEY" not in os.environ:
    # OR set the key here as a variable
    openai.api_key = ""
    
assert len(openai.Model.list()["data"]) > 0

def embed_func(c):    
    rs = openai.Embedding.create(input=c, engine="text-embedding-ada-002")
    return [record["embedding"] for record in rs["data"]]

import lancedb
from lancedb.embeddings import with_embeddings

# data = with_embeddings(embed_func, df, show_progress=True)
# data.to_pandas().head(1)

db = lancedb.connect("/tmp/lancedb")
# tbl = db.create_table("youtube-chatbot", data)
# get table
tbl = db.open_table("youtube-chatbot")

#print the length of the table
print(len(tbl))

tbl.to_pandas().head(1)

def create_prompt(query, context):
    limit = 3750

    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(context)):
        if len("\n\n---\n\n".join(context.text[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(context.text[:i-1]) +
                prompt_end
            )
            break
        elif i == len(context)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(context.text) +
                prompt_end
            )    
    print ( "prompt:", prompt )
    return prompt

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()


query = ("How do I use the Pandas library to create embeddings?")

# Embed the question
emb = embed_func(query)[0]

# Use LanceDB to get top 3 most relevant context
context = tbl.search(emb).limit(3).to_df()

# Get the answer from completion API
prompt = create_prompt(query, context)
print( "context:", context )
print ( complete( prompt ))