from utils.config import cache_uri
from context import cache, client,EMBEDDING_MODEL,NAMESPACE_UUID,chat_history_repo
import os
import json
import openai
import uuid
import hashlib

from chromadb.utils import embedding_functions
from utils.config import switch_to_ernie

openai_ef=embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name=EMBEDDING_MODEL
)

def get_collection(collection_name):
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space":"cosine"}, 
        embedding_function=openai_ef
        )

#so far the cache is used to serve the chat memory
#the format will be JSONP

def reset_cache():
    cache = {}

def reset_blob(key, on_fly = True):
    if on_fly is False:
        blob_path = os.path.join(cache_uri, key)
        if os.path.exists(blob_path):
            os.remove(blob_path)
    if key in  cache:
        del cache[key]

def get_blob_by_key(key):
    ret = cache.get(key)
    if ret is None:
        #try to load from blob
        blob_path = os.path.join(cache_uri, key)
        if os.path.exists(blob_path):
            with open(blob_path, 'rb') as f:
                ret = f.read()
                cache[key] = ret
        return ret
    else:
        return None


def get_chat_memory(username, profilename):
    key = generate_key_for_chat_memory(username, profilename)
    ret = cache.get(key)
    if ret is None:
        #try to load from blob
        blob_path = os.path.join(cache_uri, key)
        if os.path.exists(blob_path):
            with open(blob_path, 'rb') as f:
                ret =[json.loads(line) for line in f.readlines()]
                cache[key] = ret
 
    return ret
    
def set_chat_memory(username, profilename, json_list):
    key = generate_key_for_chat_memory(username, profilename)
    blob_path = os.path.join(cache_uri, key)
    if os.path.exists(blob_path):
        mode = 'a'
    else:
        mode = 'w'
    with open(blob_path, mode) as f:
        for item in json_list:
            f.write(json.dumps(item))
            f.write("\n")
    reset_blob(key)
    return get_chat_memory(username, profilename)

def clear_longterm_memroy(username, profilename):
    collection_name =generate_key_for_chat_memory(username, profilename)
    try:
        client.delete_collection(collection_name)
    except ValueError as e:
        print('collection not found for clear long term memory:',username, profilename)

def save_longterm_memory_by_datetime(username, profilename, created_at):
    history = chat_history_repo.get_chat_history_by_name_before(username, profilename, created_at)
    if history == []:
        return
    save_longterm_memory(username, profilename, history)
    chat_history_repo.set_saved_flag_by_name_and_before(username, profilename, created_at)

def save_longterm_memory(username, profilename, chat_history):
    if chat_history == []:
        return
    collection_name =generate_key_for_chat_memory(username, profilename)
    collection = get_collection(collection_name)
    id_prefix = uuid.uuid4()
    documents = []
    metadatas = []
    ids = []
    index = 0
    embeddings = []
    for chat in chat_history:
        documents.append(chat['content'])
        metadatas.append({"role":chat['role']})
        ids.append(f"{id_prefix}-{index}")
        index = index+1
        if 'embedding' in chat:
            embeddings.append(chat['embedding'])
    if embeddings == []:
        embeddings = None
    collection.add(
        embeddings=embeddings,
        documents = documents,
        metadatas = metadatas,
        ids=ids
    )
def get_longterm_memory(username, profilename, message):
    collection_name =generate_key_for_chat_memory(username, profilename)
    collection = get_collection(collection_name)
    result = collection.query(
        query_texts=message,
        n_results=6
    )
    if switch_to_ernie:
        content = ""
        for index, d in enumerate(result['documents'][0]):
            if result['metadatas'][0][index]['role']=='user':
                content += "用户提问：" + d +"\n\n"
            else:
                content += "你回答：" + d +"\n\n"
            
        return [{'role':'system','content':content}] 
    else:
        return [{'role':result['metadatas'][0][index]['role'],'content':d} for index, d in enumerate(result['documents'][0])]
    

def generate_key_for_chat_memory(username, profilename):
    raw_key = f"{username}-{profilename}"
    key = uuid.uuid3(NAMESPACE_UUID, raw_key)
    return f"ltm-{key}"