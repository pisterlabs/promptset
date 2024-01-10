from langchain.embeddings.openai import OpenAIEmbeddings
from models.databases.supabase.supabase import SupabaseDB
from models.databases.qdrant.qdrant import QdrantDB
from pydantic import BaseSettings
from supabase.client import Client, create_client
from qdrant_client import QdrantClient
from vectorstore.supabase import SupabaseVectorStore


class BrainRateLimiting(BaseSettings):
    max_brain_size: int = 52428800
    max_brain_per_user: int = 69


class BrainSettings(BaseSettings):
    openai_api_key: str
    anthropic_api_key: str
    supabase_url: str
    supabase_service_key: str
    pg_database_url: str = "not implemented"
    resend_api_key: str = "null"
    resend_email_address: str = "brain@mail.quivr.app"

class DatabaseSettings(BaseSettings):
    qdrant_location: str
    qdrant_port: int
    encoder_model: str

class LLMSettings(BaseSettings):
    private: bool = False
    model_path: str = "./local_models/ggml-gpt4all-j-v1.3-groovy.bin"


def get_supabase_client() -> Client:
    settings = BrainSettings()  # pyright: ignore reportPrivateUsage=none
    supabase_client: Client = create_client(
        settings.supabase_url, settings.supabase_service_key
    )
    return supabase_client

def get_qdrant_client() -> QdrantClient:
    settings = DatabaseSettings()  # pyright: ignore reportPrivateUsage=none
    qdrant_client: QdrantClient = QdrantClient(
        settings.qdrant_location, port=settings.qdrant_port
    )
    return qdrant_client

def get_supabase_db() -> SupabaseDB:
    supabase_client = get_supabase_client()
    return SupabaseDB(supabase_client)

def get_qdrant_db() -> QdrantDB:
    qdrant_client = get_qdrant_client()
    return QdrantDB(qdrant_client)

def get_embeddings() -> OpenAIEmbeddings:
    settings = BrainSettings()  # pyright: ignore reportPrivateUsage=none
    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key
    )  # pyright: ignore reportPrivateUsage=none
    return embeddings


def get_documents_vector_store() -> SupabaseVectorStore:
    settings = BrainSettings()  # pyright: ignore reportPrivateUsage=none
    embeddings = get_embeddings()
    supabase_client: Client = create_client(
        settings.supabase_url, settings.supabase_service_key
    )
    documents_vector_store = SupabaseVectorStore(
        supabase_client, embeddings, table_name="vectors"
    )
    return documents_vector_store
