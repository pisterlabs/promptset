url = 'redis://default:izDy3F5WHCOONRsT0i49QP1rv7gJLowa@redis-19938.c8.us-east-1-4.ec2.cloud.redislabs.com:19938'

host = 'redis-19938.c8.us-east-1-4.ec2.cloud.redislabs.com'
password = 'izDy3F5WHCOONRsT0i49QP1rv7gJLowa'
port = 19938

import gzip
import json
import pandas as pd
import numpy as np

import redis
from langchain.vectorstores.redis import Redis

r = redis.Redis(
  host=host,
  port=port,
  password=password)

r.ping()

def get_data(asin):
    data = []
    with gzip.open('AMAZON_FASHION.json.gz') as f:
        for l in f:
            data.append(json.loads(l.strip()))

    df = pd.DataFrame.from_dict(data)
    df = df[df['reviewText'].notna()]
    df = df.loc[df['asin'] == asin].copy()
    return df

df=get_data('B000KPIHQ4').reset_index()[['overall','asin','reviewText','summary','reviewerID']]

max_text_length=400
def truncate_review(text):
    return text[:max_text_length]
df['reviewText']=df.apply(lambda row: truncate_review(row['reviewText']),axis=1)

df['overall']=df.apply(lambda row: int(row['overall']),axis=1)

from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

reviews=df.head(100).to_dict(orient='index')

texts=df.head(100)['reviewText'].to_list()
vectors=embeddings.embed_documents(texts)

summaries=df.head(100)['summary'].to_list()
summary_vectors=embeddings.embed_documents(summaries)

print(len(vectors[0]))

r.execute_command('FT._LIST')

from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField
from redis.commands.search.field import NumericField

from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from redis.commands.search.query import Query

pipe = r.pipeline(transaction=False)

schema = (
    TextField("overall"),
    TextField("reviewText"),
    TextField("summary"),
    VectorField("vector", "HNSW", {"TYPE": "FLOAT32", "DIM": 768, "DISTANCE_METRIC": "COSINE"}),
)

prefix="doc:reviewidx_hash"

r.ft("reviewidx_hashed").create_index(fields=schema,
                                      definition=IndexDefinition(prefix=[prefix],index_type=IndexType.HASH)
                                     )

for i in reviews.keys():
    key=prefix+':' + reviews[i]['reviewerID']
    record = reviews[i]
    record['vector']=np.array(vectors[i]).astype(np.float32).tobytes()
    pipe.hset(key,mapping=record)
pipe.execute()


r.execute_command('FT._LIST')

