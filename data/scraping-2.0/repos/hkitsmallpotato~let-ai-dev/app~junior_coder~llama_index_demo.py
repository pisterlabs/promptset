from custom_llm import CustomLLM

import chromadb
from chromadb.config import Settings

from llama_index.vector_stores import ChromaVectorStore

from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    LangchainEmbedding,
    ServiceContext,
    StorageContext
)

#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings 

chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="chroma_store"
    #persist_directory="/path/to/persist/directory" # Optional, defaults to .chromadb/ in the current directory
))

chroma_collection = chroma_client.create_collection(name="llama_vec_store")
#No embedding_function provided, using default embedding function: DefaultEmbeddingFunction https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# construct vector store
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection,
)

# construct vector store and customize storage context
storage_context = StorageContext.from_defaults(
    vector_store = vector_store,
    #embed_model = embed_model
)

# Load your documents
documents = SimpleDirectoryReader('data').load_data()

# define LLM
llm_predictor = LLMPredictor(llm=CustomLLM())
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model = embed_model
)


# Create the index
index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

# Create the query engine
query_engine = index.as_query_engine()

# Query the engine
response = query_engine.query("How does langchain enable agent use case?")

