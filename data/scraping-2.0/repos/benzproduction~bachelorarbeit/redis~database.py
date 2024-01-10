import pandas as pd 
import numpy as np
from redis import Redis
from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField, NumericField
from redis.commands.search.query import Query
from openai.embeddings_utils import get_embedding
from typing import List
from models import DocumentChunk
from config import EMBEDDINGS_MODEL, PREFIX, VECTOR_FIELD_NAME

# Get a Redis connection
def get_redis_connection(password=None,host='localhost',port='6379',db=0):
    r = Redis(host=host, port=port, db=db,decode_responses=False, password=password)
    return r

# Create a Redis index to hold our data
def create_hnsw_index (redis_conn,vector_field_name,vector_dimensions=1536, distance_metric='COSINE'):
    redis_conn.ft().create_index([
        VectorField(vector_field_name, "HNSW", {"TYPE": "FLOAT32", "DIM": vector_dimensions, "DISTANCE_METRIC": distance_metric}),
        TextField("filename"),
        TextField("text_chunk"),        
        NumericField("file_chunk_index")
    ])

# Create a Redis pipeline to load all the vectors and their metadata
def load_vectors(client:Redis, input_list, vector_field_name):
    p = client.pipeline(transaction=False)
    for text in input_list:    
        #hash key
        key=f"{PREFIX}:{text['id']}"
        
        #hash values
        item_metadata = text['metadata']
        #
        item_keywords_vector = np.array(text['vector'],dtype= 'float32').tobytes()
        item_metadata[vector_field_name]=item_keywords_vector
        
        # HSET
        p.hset(key,mapping=item_metadata)
            
    p.execute()

def save_chunks(r:Redis, vectors: List[DocumentChunk], index: str) -> None:
    """
    Saves the vectors to Redis
    """
    assert r.ping(), "Redis is not connected"
    try:
        print(f"Docs in index: {r.ft(index).info()['num_docs']}")
    except Exception as e:
        print(f"Index {index} does not exist. Exiting")
        exit(1)
    p = r.pipeline(transaction=False)
    # load vectors
    for vector in vectors:
        #hash key
        key=f"{index}:{vector.id}"
        item_metadata = {}
        item_metadata["filename"] = vector.metadata.source_filename
        item_metadata["text_chunk"] = vector.text
        item_metadata["page"] = vector.metadata.page
        item_keywords_vector = np.array(vector.embedding,dtype= 'float32').tobytes()
        item_metadata[VECTOR_FIELD_NAME]=item_keywords_vector

        # HSET
        r.hset(key,mapping=item_metadata)
    p.execute()

# Make query to Redis
def query_redis(redis_conn,query,index_name, top_k=5):
    
    

    ## Creates embedding vector from user query
    # embedded_query = np.array(openai.Embedding.create(
    #                                             input=query,
    #                                             model=EMBEDDINGS_MODEL,
    #                                         )["data"][0]['embedding'], dtype=np.float32).tobytes()
    
    # above code has to rewritten to use the azure openai services
    embedded_query = np.array(get_embedding(query, engine = 'text-embedding-ada-002'), dtype=np.float32).tobytes()



    #prepare the query
    q = Query(f'*=>[KNN {top_k} @{VECTOR_FIELD_NAME} $vec_param AS vector_score]').sort_by('vector_score').paging(0,top_k).return_fields('vector_score','filename','text_chunk','text_chunk_index').dialect(2) 
    params_dict = {"vec_param": embedded_query}

    
    #Execute the query
    results = redis_conn.ft(index_name).search(q, query_params = params_dict)
    
    return results

# Get mapped documents from Weaviate results
def get_redis_results(redis_conn,query,index_name):
    
    # Get most relevant documents from Redis
    query_result = query_redis(redis_conn,query,index_name)
    
    # Extract info into a list
    query_result_list = []
    for i, result in enumerate(query_result.docs):
        result_order = i
        text = result.text_chunk
        score = result.vector_score
        query_result_list.append((result_order,text,score))
        
    # Display result as a DataFrame for ease of us
    result_df = pd.DataFrame(query_result_list)
    result_df.columns = ['id','result','certainty',]
    return result_df

def get_redis_results2(redis_conn,query,index_name, top_k=5):
    
    # Get most relevant documents from Redis
    query_result = query_redis(redis_conn,query,index_name, top_k=top_k)
    # if the result is empty, return an empty dataframe
    if query_result.total == 0:
        return pd.DataFrame()
    
    # Extract info into a list
    query_result_list = []
    for i, result in enumerate(query_result.docs):
        result_order = i
        text = result.text_chunk
        score = result.vector_score
        filename = result.id
        query_result_list.append((result_order,text,score, filename))
        
    # Display result as a DataFrame for ease of us
    result_df = pd.DataFrame(query_result_list)
    result_df.columns = ['id','result','certainty', 'filename']
    return result_df