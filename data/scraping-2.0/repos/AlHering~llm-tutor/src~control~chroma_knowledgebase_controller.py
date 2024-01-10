# -*- coding: utf-8 -*-
"""
****************************************************
*                    LLM Tutor                     *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from typing import Any, List
from chromadb.api.types import EmbeddingFunction, Embeddings, Documents
from chromadb.config import Settings
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from src.configuration import configuration as cfg
from src.utility.bronze.hashing_utility import hash_text_with_sha256
from src.model.knowledgebase_control.abstract_knowledgebase_controller import KnowledgeBaseController
from src.utility.silver import embedding_utility


class ChromaKnowledgeBase(KnowledgeBaseController):
    """
    Class for handling knowledge base interaction with ChromaDB.
    """

    def __init__(self, peristant_directory: str, metadata: dict = None, base_embedding_function: EmbeddingFunction = None) -> None:
        """
        Initiation method.
        :param peristant_directory: Persistant directory for ChromaDB data.
        :param metadata: Embedding collection metadata. Defaults to None.
        :param base_embedding_function: Embedding function for base collection. Defaults to T5 large.
        """
        if not os.path.exists(peristant_directory):
            os.makedirs(peristant_directory)
        self.peristant_directory = peristant_directory
        self.base_embedding_function = embedding_utility.LocalHuggingFaceEmbeddings(
            cfg.PATHS.INSTRUCT_XL_PATH
        ) if base_embedding_function is None else base_embedding_function
        self.client_settings = Settings(persist_directory=peristant_directory,
                                        chroma_db_impl='duckdb+parquet')

        self.databases = {}
        self.base_chromadb = self.get_or_create_collection("base")

    # Override
    def get_or_create_collection(self, name: str, metadata: dict = None, embedding_function: EmbeddingFunction = None) -> Chroma:
        """
        Method for retrieving or creating a collection.
        :param name: Collection name.
        :param metadata: Embedding collection metadata. Defaults to None.
        :param embedding_function: Embedding function for the collection. Defaults to base embedding function.
        :return: Database API.
        """
        if name not in self.databases:
            self.databases[name] = Chroma(
                persist_directory=self.peristant_directory,
                embedding_function=self.base_embedding_function if embedding_function is None else embedding_function,
                collection_name=name,
                collection_metadata=metadata,
                client_settings=self.client_settings
            )
        return self.databases[name]

    # Override
    def get_retriever(self, name: str, search_type: str = "similarity", search_kwargs: dict = {"k": 4, "include_metadata": True}) -> VectorStoreRetriever:
        """
        Method for acquiring a retriever.
        :param name: Collection to use.
        :param search_type: The retriever's search type. Defaults to "similarity".
        :param search_kwargs: The retrievery search keyword arguments. Defaults to {"k": 4, "include_metadata": True}.
        :return: Retriever instance.
        """
        db = self.databases.get(name, self.databases["base"])
        search_kwargs["k"] = min(
            search_kwargs["k"], len(db.get()["ids"]))
        return db.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    # Override
    def embed_documents(self, name: str, documents: List[Document], ids: List[str] = None) -> None:
        """
        Method for embedding documents.
        :param name: Collection to use.
        :param documents: Documents to embed.
        :param ids: Custom IDs to add. Defaults to the hash of the document contents.
        """
        self.databases[name].add_documents(documents=documents, ids=[
            hash_text_with_sha256(document.page_content) for document in documents] if ids is None else ids)
        self.databases[name].persist()
