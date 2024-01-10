from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import VectorStore

from app.config.conf import MILVUS_CONNECTION_ARGS, EMBEDDING_DEPLOYMENT_NAME, DEFAULT_TEXT_FIELD
from app.milvus.milvus import Milvus


def get_vector_store(
        collection_name: str
) -> VectorStore:
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_DEPLOYMENT_NAME,
        openai_api_version="2020-11-07",
        chunk_size=3000)

    qa_vector_store = Milvus.from_existing(
        embedding=embeddings,
        connection_args=MILVUS_CONNECTION_ARGS,
        collection_name=collection_name,
        text_field=DEFAULT_TEXT_FIELD
    )
    return qa_vector_store
