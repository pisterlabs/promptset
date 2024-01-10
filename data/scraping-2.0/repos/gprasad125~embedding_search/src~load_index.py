import openai
import pinecone
import os 

from dotenv import load_dotenv, find_dotenv

def create_index(): 
    load_dotenv(find_dotenv())

    openai.api_key = os.environ.get('OPENAI_API_KEY')
    pinecone.init(api_key = os.environ.get('PINECONE_API_KEY'), environment = os.environ.get('PINECONE_ENV'))

    index_name = 'quads'
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)
    else:
        pinecone.delete_index(index_name)
        pinecone.create_index(index_name, dimension=1536)

    index = pinecone.Index(index_name)
    return index 