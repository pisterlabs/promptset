url = 'redis://default:izDy3F5WHCOONRsT0i49QP1rv7gJLowa@redis-19938.c8.us-east-1-4.ec2.cloud.redislabs.com:19938'

host = 'redis-19938.c8.us-east-1-4.ec2.cloud.redislabs.com'
password = 'izDy3F5WHCOONRsT0i49QP1rv7gJLowa'
port = 19938

import redis
from langchain.vectorstores.redis import Redis

r = redis.Redis(
  host=host,
  port=port,
  password=password)

r.ping()

r.keys()

import gzip
import json
import pandas as pd

def get_data(asin):
    data = []
    with gzip.open('AMAZON_FASHION.json.gz') as f:
        for l in f:
            data.append(json.loads(l.strip()))

    df = pd.DataFrame.from_dict(data)
    df = df[df['reviewText'].notna()]
    df = df.loc[df['asin'] == asin].copy()
    return df

df=get_data('B000KPIHQ4')[['overall','asin','reviewText','summary']]
df1 = df[:5]


from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

reviews=df.head(2).to_dict(orient='index')
print(reviews)

texts=df.head(2)['reviewText'].to_list()
vectors=embeddings.embed_documents(texts)


summaries=df.head(2)['summary'].to_list()
summary_vectors=embeddings.embed_documents(summaries)

metadata=[dict(rating=i) for i in df1['overall'].tolist()]
print(metadata[:2])

vstore = Redis.from_texts(texts, embeddings, redis_url=url,  index_name='reviewidx')

vstore_meta = Redis.from_texts(texts, embeddings, redis_url=url, index_name='reviewidx_meta',metadatas=metadata)

# print(r.keys)

# r.execute_command('FT._LIST')

# vec=r.hget('doc:reviewidx_meta:84187fccf83f4493968b663b26360b7e','content_vector')
#
# import numpy as np
#
# print(np.frombuffer(vec, dtype=np.float32))