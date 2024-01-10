import os
from pathlib import Path
import logging
from brics_tools.utils import helper
from llama_index import (
    LangchainEmbedding,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    set_global_service_context,
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.vector_stores import ChromaVectorStore
from llama_index.callbacks import CallbackManager, OpenInferenceCallbackHandler
from llama_index.schema import MetadataMode

import chromadb
from chromadb.config import Settings


class StudyInfoVectorStoreIndexManager:
    def __init__(self, config, nodes=None):
        self.config = config
        self.nodes = nodes
        self.indices = {}
        self.init_callback_manager()

    def init_callback_manager(self):
        self.callback_handler = OpenInferenceCallbackHandler()
        self.callback_manager = CallbackManager([self.callback_handler])

    def init_vectorstore(self):
        storage_path_root = self.config.storage_context.storage_path_root
        self.storage_path_root = storage_path_root

        # Initialize ChromaDB client
        distance_metric = dict(self.config.collections.vectordb.distance_metric)
        self.client = chromadb.PersistentClient(
            path=storage_path_root, settings=Settings(anonymized_telemetry=False)
        )

        # Initialize Chroma collection and VectorStore
        chroma_collection = self.client.get_or_create_collection(
            self.config.index_id, metadata=distance_metric
        )
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    def init_service_context(self):
        # Get values from YAML configuration
        model_name = self.config.collections.embed.model_name
        encode_kwargs = self.config.collections.embed.encode_kwargs
        batch_size = self.config.collections.embed.model_kwargs.batch_size

        # Initialize Embedding Model
        self.embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={},
                encode_kwargs=encode_kwargs,
            ),
            embed_batch_size=batch_size,  # Override defaults
        )

        # Initialize ServiceContext
        self.service_context = ServiceContext.from_defaults(
            llm=None,
            embed_model=self.embed_model,
            callback_manager=self.callback_manager,
        )

    def init_storage_context(self):
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def load_vectorstore_index(self):
        self.init_vectorstore()
        self.init_service_context()
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, service_context=self.service_context
        )

    def create_index(self):
        if self.nodes is None:
            raise ValueError(
                "The 'nodes' parameter must be provided to create a new index."
            )

        # Initialize ServiceContext and StorageContext here
        self.init_vectorstore()
        self.init_service_context()
        self.init_storage_context()

        # Now proceed with creating the index
        self.index = VectorStoreIndex(
            self.nodes,
            storage_context=self.storage_context,
            service_context=self.service_context,
            show_progress=True,
        )

        # Persist index to local storage #TODO: Separate this out in future in case want to just use in memory or persist in another fashion
        if not os.path.exists(self.storage_path_root):
            os.makedirs(self.storage_path_root)
        self.index.storage_context.persist(self.storage_path_root)
