import os
from dotenv import load_dotenv

from llama_index import (
    GPTVectorStoreIndex,
    GPTSimpleKeywordTableIndex, 
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext
)
from llama_index.vector_stores import PineconeVectorStore
# from langchain.llms.openai import OpenAIChat
from langchain.chat_models import ChatOpenAI
from pathlib import Path

import pinecone
from llama_index.storage.storage_context import StorageContext
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
# Creating a Pinecone index
api_key = os.getenv("PINECONE_API_KEY") 
pinecone.init(api_key=api_key, environment="northamerica-northeast1-gcp")
index = pinecone.Index("openai")

# can define filters specific to this vector index (so you can
# reuse pinecone indexes)
metadata_filters = {"company_title_document": "meta_10K"}

# construct vector store
vector_store = PineconeVectorStore(
    pinecone_index=index,
    metadata_filters=metadata_filters
)

# LLM Predictor (gpt-3.5-turbo)
llm_predictor_chatgpt = LLMPredictor(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, chunk_size_limit=512)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

meta_index = GPTVectorStoreIndex.from_documents(
    [], storage_context=storage_context, service_context=service_context
)

index = meta_index.as_query_engine(service_context=service_context)
response = index.query(
    "What is the revenue of 'Family of Apps' division of Meta in 2022?"
)
