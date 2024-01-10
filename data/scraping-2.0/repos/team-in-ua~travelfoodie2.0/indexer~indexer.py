import openai 
import os
from dotenv import load_dotenv 
import pinecone
import pandas as pd
from tqdm.auto import tqdm

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
model = os.environ.get("MODEL")

index_name = os.environ.get("PINECONE_INDEX_NAME")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")   
pinecone_env = os.environ.get("PINECONE_ENV")

data = pd.read_csv('data.csv')

pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_env
)

pinecone.delete_index(index_name)

# check if 'openai' index already exists (only create index if not)
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)
# connect to index
index = pinecone.Index(index_name)


count = 0  # we'll use the count to create unique IDs
batch_size = 10  # process everything in batches of 32
for i in tqdm(range(0, len(data['text']), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(data['text']))
    # get batch of lines and IDs
    lines_batch = data['text'][i: i+batch_size].tolist()
    countries_batch = data['country'][i: i+batch_size].tolist()
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = openai.Embedding.create(input=lines_batch, engine=model)
    embeds = [record['embedding'] for record in res['data']]
    # prep metadata and upsert batch
    meta = [{'text': line, 'country': country} for line, country in zip(lines_batch,countries_batch)]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))


# query = "I like kimchi"

# # create the query embedding
# xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']

# # query, returning the top 5 most similar results
# res = index.query([xq], top_k=30, include_metadata=True)

# for match in res['matches']:
#     print(f"{match['score']:.5f}: {match['metadata']['country']}")