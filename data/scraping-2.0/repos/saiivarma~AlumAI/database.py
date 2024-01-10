import os
import pinecone
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from LLM import get_embedding

def init(API_key, env):
    """
    Initializing pinecone
    """
    pinecone.init(api_key = API_key, environment = env)

    active_indexes = pinecone.list_indexes()

    if active_indexes == []:
        pinecone.create_index("alumai", dimension=1536, metric = 'cosine' )

    return pinecone.list_indexes()

def data_loader(file):
    return UnstructuredPDFLoader(file).load()

    
def load_data(data_path):

    data = {}
  
    files = os.listdir(data_path)

    for i in files:
        data[i] = data_loader("{}/{}".format(data_path,i))
        
    return data

def create_vectors(data, category):

    embeddings = []
    for i in data.keys():
        embeddings.append((i,get_embedding(data[i][0].page_content),{'source':data[i][0].metadata['source'],'category':category}))
    
    return embeddings

def insert(index_name, embeddings):

    index = pinecone.Index(index_name)
    status = index.upsert(embeddings)

    return status

def similarity_search(index_name, query, category):

    index = pinecone.Index(index_name)
    results = index.query(
    vector=get_embedding(query),
    filter={
        "category": {"$eq": category}
    },
    top_k=2,
    include_metadata=True
    )
    ids = []
    for i in results['matches']:
        ids.append(i['id'])

    return ids

def fetch_data(ids, data_path):
    
    data = []
    for i in ids:
        data.append(data_loader("{}/{}".format(data_path,i))[0])

    return data


