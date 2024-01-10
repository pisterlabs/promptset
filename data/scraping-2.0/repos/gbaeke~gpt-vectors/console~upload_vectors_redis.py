import feedparser
import os
import numpy as np
import openai
import requests
from bs4 import BeautifulSoup
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Redis connection details
redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')
redis_password = os.getenv('REDIS_PASSWORD')

# Connect to the Redis server
conn = redis.Redis(host=redis_host, port=redis_port, password=redis_password, encoding='utf-8', decode_responses=True)


SCHEMA = [
    TextField("url"),
    VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"}),
]

# Create the index
try:
    conn.ft("posts").create_index(fields=SCHEMA, definition=IndexDefinition(prefix=["post:"], index_type=IndexType.HASH))
except Exception as e:
    print("Index already exists")
    

# URL of the RSS feed to parse
url = 'https://blog.baeke.info/feed/'

# Parse the RSS feed with feedparser
feed = feedparser.parse(url)

# get number of entries in feed
entries = len(feed.entries)
print("Number of entries: ", entries)

p = conn.pipeline(transaction=False)
for i, entry in enumerate(feed.entries[:50]):
    # report progress
    print("Create embedding and save for entry ", i, " of ", entries)

    r = requests.get(entry.link)
    soup = BeautifulSoup(r.text, 'html.parser')
    article = soup.find('div', {'class': 'entry-content'}).text

    # vectorize with OpenAI text-emebdding-ada-002
    embedding = openai.Embedding.create(
        input=article,
        model="text-embedding-ada-002"
    )

    # print the embedding (length = 1536)
    vector = embedding["data"][0]["embedding"]

    print(embedding)
    exit()

    # convert to numpy array
    vector = np.array(vector).astype(np.float32).tobytes()

    # Create a new hash with the URL and embedding
    post_hash = {
        "url": entry.link,
        "embedding": vector
    }

    
    
    # add_document() is deprecated
    conn.hset(name=f"post:{i}", mapping=post_hash)

p.execute()

print("Vector upload complete.")
