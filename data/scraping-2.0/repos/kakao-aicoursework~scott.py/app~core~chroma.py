from typing import Optional, Any

from rxconfig import config
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.schema.retriever import BaseRetriever
from langchain.vectorstores import Chroma

CLIENT: Optional[chromadb.Client] = None
DB: Optional[Chroma] = None
RETRIEVER: Optional[BaseRetriever] = None


def init_chroma():
    global CLIENT, DB, RETRIEVER
    if not DB:
        DB = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIRECTORY,
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(),
            collection_metadata={"hnsw:space": "cosine"}
        )
        CLIENT = DB._client
        RETRIEVER = DB.as_retriever()


def get_similar_docs(
    query: str,
    top_k: int = 5,
    filters: Optional[dict[str, Any]] = None,
    only_contents: bool = True,
) -> list[str | Document]:
    global DB
    if not DB:
        raise Exception("CHROMA Retriever is not initialized")
    docs = DB.similarity_search(query, k=top_k, filter=filters)
    if only_contents:
        return [doc.page_content for doc in docs]
    return docs


def query_db(
    query: str,
    metadata: Optional[dict[str, Any]] = None,
    only_contents: bool = True
) -> list[str | Document]:
    global RETRIEVER
    if not RETRIEVER:
        raise Exception("CHROMA Retriever is not initialized")
    docs = RETRIEVER.get_relevant_documents(query, metadata=metadata)
    if only_contents:
        return [doc.page_content for doc in docs]
    return docs
