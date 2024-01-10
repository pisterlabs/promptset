from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.redis import Redis
from langchain.embeddings import HuggingFaceEmbeddings


INDEX_NAME = "redis_snowflake"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def upload(documents: list[Document]) -> None:
    Redis.from_documents(
        documents, embeddings, redis_url="redis://localhost:6379", index_name=INDEX_NAME
    )


def create() -> VectorStore:
    return Redis.from_existing_index(
        embeddings, redis_url="redis://localhost:6379", index_name=INDEX_NAME
    )
