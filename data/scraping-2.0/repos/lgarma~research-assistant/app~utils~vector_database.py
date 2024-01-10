"""Vector database utilities."""
import streamlit as st
from langchain.embeddings.cache import CacheBackedEmbeddings, _create_key_encoder
from langchain.schema import Document
from langchain.storage import LocalFileStore
from langchain.vectorstores import Milvus

state = st.session_state


def connect_to_vector_db():
    """Connect to pre-existing vector database.

    Sets up a cached embedder to save the embeddings
    The cached documents prevent duplicated documents.
    """
    state["nice_collection_name"] = state["collection_name"].replace("_", " ").title()
    state["cache"] = LocalFileStore(f"./cache/{state['collection_name']}")
    state["cached_embedder"] = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=state["embedding_model"],
        document_embedding_cache=state["cache"],
        namespace=state["collection_name"],
    )
    state["vector_db"] = Milvus(
        collection_name=state["collection_name"],
        embedding_function=state["cached_embedder"],
    )


def disconnect_from_vector_db():
    """Disconnect from the vector database."""
    if "vector_db" in state:
        del state["vector_db"]

    if "cached_embedder" in state:
        del state["cached_embedder"]

    if "cache" in state:
        del state["cache"]

    if "collection_name" in state:
        del state["collection_name"]

    for key in state:
        if key.startswith("batch"):
            del state[key]


def cache_documents_embeddings(docs: list[Document]):
    """Get the embeddings for the documents and cache them.

    Checks for duplicates before caching.
    Returns list of documents that were not in the database.
    """
    st.write("Checking cache for duplicates...")
    new_docs = check_for_duplicates(docs)
    if len(new_docs) > 0:
        st.write(f"Encoding {len(new_docs)} new documents.")
        state["cached_embedder"].embed_documents([doc.page_content for doc in new_docs])
    return new_docs


def check_for_duplicates(docs: list[Document]):
    """Only return documents that where not in the database.

    Extract the uuid of the documents. If the uuid is not in the database, return
    the document.
    """
    # encoder used by langchain
    preexisting_keys = list(state["cache"].yield_keys())
    key_encoder = _create_key_encoder(namespace=state["collection_name"])
    keys = [key_encoder(doc.page_content) for doc in docs]
    return [doc for doc, key in zip(docs, keys) if key not in preexisting_keys]


def get_all_documents() -> list[Document]:
    """Get all documents from the vector database that is currently loaded"""
    return state["vector_db"].similarity_search(
        query="query",
        k=state["vector_db"].col.num_entities,
        param={"metric_type": "L2", "params": {"nprobe": 16}},
    )
