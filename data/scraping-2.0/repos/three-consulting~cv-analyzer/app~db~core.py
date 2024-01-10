import chromadb
from chromadb.config import Settings
from config import Settings as AppSettings
from langchain.embeddings import OpenAIEmbeddings

client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=AppSettings().chroma_path,
    )
)

embedding_function = OpenAIEmbeddings(
    openai_api_key=AppSettings().openai_api_key,
    model=AppSettings().openai_embedding_model,
).embed_documents
