import chromadb
from langchain.vectorstores import Chroma
from umbertobot import indexing
from umbertobot.models import EmbeddingConfig, PersistenceConfig
from typing import Union, PathLike


def get_semantic_search_engine(
    persistence_config: Union[PersistenceConfig, PathLike],
    embedding_config: Union[EmbeddingConfig, PathLike],
):
    embedding_config = (
        embedding_config
        if type(embedding_config) is EmbeddingConfig
        else indexing.EmbeddingConfig.load_from_yaml(embedding_config)
    )
    persistence_config = (
        persistence_config
        if type(persistence_config) is PersistenceConfig
        else indexing.PersistenceConfig.load_from_yaml(persistence_config)
    )
    embeddings = embedding_config.load_embeddings()
    chroma_settings = chromadb.config.Settings(
        persist_directory=persistence_config.persistence_directory,
        chroma_db_impl="duckdb+parquet",
    )
    return Chroma(
        collection_name=persistence_config.collection_name,
        embedding_function=embeddings,
        client_settings=chroma_settings,
    )
