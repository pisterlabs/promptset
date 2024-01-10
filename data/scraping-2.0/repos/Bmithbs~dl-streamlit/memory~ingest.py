"""This is the logic for ingesting data into LangChain."""
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import time    
import os

def ingest(memory_path="raw_memory/lsc.txt", database_path="./", key=None):
    embedding = OpenAIEmbeddings(openai_api_key=key)

    loader = TextLoader(memory_path) # source
    documents = loader.load()
    # Here we split the documents, as needed, into smaller chunks.
    # We do this due to the context limits of the LLMs.
    text_splitter = CharacterTextSplitter(chunk_size=500, separator="\n")
    docs = text_splitter.split_documents(documents)
    persist_directory=database_path
    print("Start to ingest!")
    start_time = time.time()
    db = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
    print(f"Ingestion complete!, time cost: {time.time() - start_time} seconds.")
    db.persist()


class IngestMemoey():
    def __init__(self, memory_path="raw_memory/lsc.txt", database_path="./", openai_api_key=None):
        
        # 传参
        self.memory_path = memory_path
        self.database_path = database_path
        self.key = openai_api_key

        self.embedding = OpenAIEmbeddings(openai_api_key=self.key)
        self.loader = TextLoader(self.memory_path) # source
        self.documents = self.loader.load() # 加载文件

        # Here we split the documents, as needed, into smaller chunks.
        # We do this due to the context limits of the LLMs.
        self.text_splitter = CharacterTextSplitter(chunk_size=500, separator="\n")
        self.docs = self.text_splitter.split_documents(self.documents)
        
        # 存储路径
        self.persist_directory=self.database_path


    def ingest(self):
        
        print("Start to ingest!")
        start_time = time.time()
        # vectorize the memory, then save it 
        db = Chroma.from_documents(documents=self.docs, embedding=self.embedding, persist_directory=self.persist_directory)

        print(f"Ingestion complete!, time cost: {time.time() - start_time} seconds.")
        db.persist()

if __name__ == "__main__":
    from config import OPENAI_API_KEY
    if OPENAI_API_KEY:
        print('API key found, start ingesting...')
        ingest = IngestMemoey()
        ingest(key=OPENAI_API_KEY)
    else:
        print('API key not found, please export the OpenAI API key as the environment variable')