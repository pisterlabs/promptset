from llama_index import StorageContext, load_index_from_storage
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
import os
import openai

from key1 import KEY
import os
import openai

os.environ['OPENAI_API_KEY'] = KEY

def generate_response(prompt,dir1):
    v_dir = f'./vector/{dir1}'
    print ("v_dir", v_dir)
    storage_context = StorageContext.from_defaults(persist_dir=v_dir)
    index = load_index_from_storage(storage_context)
    #print ("index", index, dir(index))
    query_engin = index.as_query_engine(similarity_top_k=2)
    question = prompt
    response = query_engin.query(question)
    print (response, type(response))
    return str(response)
