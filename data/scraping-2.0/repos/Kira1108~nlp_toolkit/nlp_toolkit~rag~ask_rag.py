from __future__ import annotations
import os

try:
    from llama_index import (
        VectorStoreIndex,
        SimpleDirectoryReader,
        StorageContext,
        load_index_from_storage
    )
except:
    pass

from dataclasses import dataclass

def get_or_create_index_local(persist_dir = './storage', documents_dir :str= "data"):
    if not os.path.exists(persist_dir):
        documents = SimpleDirectoryReader(documents_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir = persist_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        
    return index

@dataclass
class LocalDirRag:
    
    documents_dir:str = "data"
    persist_dir:str = './storage'
    
    def __post_init__(self):
        try:
            import openai
            openai.api_key = os.environ["OPENAI_API_KEY"]
        except:
            raise ValueError("OpenAI API not installed. Please install it using pip install openai")
        
        try:
            import llama_index
        except:
            raise ValueError("llama_index not installed. Please install it using pip install llama_index")
        
        if openai.api_key is None:
            raise ValueError("OpenAI API key not found. Please set it as an environment variable OPENAI_API_KEY")
        
        if len(openai.api_key) < 5:
            raise ValueError("OpenAI API key not in correct format. Please set it as an environment variable OPENAI_API_KEY")
        
        self.index = get_or_create_index_local(persist_dir = self.persist_dir, 
                                          documents_dir = self.documents_dir)
    
    def ask(self, query:str):
        engine = self.index.as_query_engine()
        return engine.query(query)
    
    @property 
    def chatbot(self):
        engine = self.index.as_chat_engine()
        return RagChatBot(engine)
    
    def __call__(self, query:str):
        return self.ask(query).response
    

class RagChatBot:
    def __init__(self, engine):
        self.engine = engine
        
    def __call__(self,query):
        return self.engine.chat(query)
    