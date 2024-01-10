#This needs to be run only once for creating a local index store

from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import os
from langchain.chat_models import ChatOpenAI

#For powershell env variable setup
#$env:OPENAI_API_KEY = "" 


def create_index(folder_path):

    documents = SimpleDirectoryReader(folder_path).load_data()
    index = VectorStoreIndex.from_documents(documents)

    # By default, data is stored in-memory. This is to persist to disk (under ./storage)
    index.storage_context.persist()

    return index 

create_index("data")
