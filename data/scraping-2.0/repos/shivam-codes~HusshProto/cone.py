import os
import pinecone
import cohere
from dotenv import load_dotenv

load_dotenv()

cohere_key = os.getenv('cohere_key')
pinecone_key = os.getenv('pinecone_key')
pinecone_env = os.getenv('pinecone_env')


co = cohere.Client(cohere_key)
pinecone.init(api_key=pinecone_key, environment=pinecone_env)


def createEmbeddings(data):
    embeds = co.embed(texts=data,model='small', truncate='LEFT').embeddings
    return embeds


def createPineconeIndex(id, shape):
    index_name = 'coherent-pinecone-' + id
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=shape,metric='cosine')
    return index_name

def getIndex(index_name):
    index = pinecone.Index(index_name)
    return index

def populateIndex(index_name, data, embeds):
    index = getIndex(index_name)
    batch_size = 200
    ids = [str(i) for i in range(len(data))]
    metadata = [{'data': d} for d in data]
    to_upsert = list(zip(ids,embeds, metadata))
    for i in range(0, len(data), batch_size):
        i_end = min(i+batch_size, len(data))
        index.upsert(vectors=to_upsert[i:i_end])
    return index.describe_index_stats()


def queryFromDatabase(index_name, query):
    index = getIndex(index_name)
    embed = createEmbeddings([query])
    res = index.query(embed, top_k=20, include_metadata=True)
    result = []
    for r in res['matches']:
        data = (r['metadata']['data'])
        result.append(data)
    return result

