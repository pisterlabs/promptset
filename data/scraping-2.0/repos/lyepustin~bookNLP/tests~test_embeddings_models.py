 # File: LangChainchatOpenAI.py
# Author: Denys L
# Date: October 8, 2023
# Description: 

from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from ebooklib import epub
from bs4 import BeautifulSoup
import qdrant_client
import ebooklib
import logging
import os
import sys
import time

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
load_dotenv()

from qdrant_client.http.models import Distance, VectorParams

def recreate_qdrant_collection(collection_name, size):

    client = get_qdrant_client()
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )
        logging.info(f"'{collection_name}' collection re-created.")
    except Exception as e:
        logging.error(
            f"on create collection '{collection_name}'. " + str(e).replace('\n', ' '))


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator=str(os.getenv("TEXT_SPLITTER_SEPARATOR")),
        chunk_size=int(os.getenv("TEXT_SPLITTER_CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("TEXT_SPLITTER_CHUNK_OVERLAP")),
        length_function=len
    )
    chunks = text_splitter.split_text(str(text))
    return chunks


def get_qdrant_client():
    return qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        port=os.getenv("QDRANT_PORT"),
        api_key=os.getenv("QDRANT_API_KEY"),
        https=True)


def get_vector_store(embedding_model):
    client = get_qdrant_client()
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
    )
    return Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )


def read_book(path):
    with open(path, 'r', encoding="utf-8") as file:
        return file.read()

def read_book_sample():
    return os.getenv("TEXT_SAMPLE")


def add_some_text():
    recreate_qdrant_collection(
        os.getenv("QDRANT_COLLECTION_NAME"), os.getenv("QDRANT_COLLECTION_SIZE"))

    # content = read_book('docs/quijote.txt')   
    content = read_book_sample() 
        
    text_chunks = get_text_chunks(content)
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
    vector_store = get_vector_store(embedding_model)
    print(len(text_chunks))
    ids = vector_store.add_texts(text_chunks)
    print(ids)
        

def make_some_query():
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=get_vector_store(embedding_model).as_retriever(),
        memory=memory,
        max_tokens_limit=16000,
        verbose=True,
    )
    response = conversation_chain({'question': "Que paso al acabar el servicio de carne?"})
    print(response)
            
# results = [
#     text-embedding-ada-002 - Dim: 1536
#     text-search-ada-doc-001 - Dim: 1024
#     text-search-babbage-doc-001 - Dim: 2048
#     text-search-curie-doc-001 - Dim: 4096
#     text-search-davinci-doc-001 - Dim: 12288
#     text-similarity-ada-001 - Dim: 1024
#     text-similarity-babbage-001 - Dim: 2048
#     text-similarity-curie-001 - Dim: 4096
#     text-similarity-davinci-001 - Dim: 12288
# ]      
        

if __name__ == '__main__':
    add_some_text()
    # make_some_query()