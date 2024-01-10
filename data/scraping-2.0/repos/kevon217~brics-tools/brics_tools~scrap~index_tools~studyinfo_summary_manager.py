import os
from pathlib import Path
import logging
from brics_tools.utils import helper
from llama_index import (
    LangchainEmbedding,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    LLMPredictor,
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.indices.loading import load_index_from_storage
from llama_index.node_parser import SimpleNodeParser
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.vector_stores import ChromaVectorStore
from llama_index.callbacks import CallbackManager, OpenInferenceCallbackHandler

import chromadb

CUSTOM_SUMMARY_QUERY = "Succinctly describe what this study is about."


class StudyInfoSummaryIndexManager:
    def __init__(self, config, docs=None):
        self.config = config
        self.docs = docs
        self.indices = {}
        self.init_callback_manager()

    def init_callback_manager(self):
        self.callback_handler = OpenInferenceCallbackHandler()
        self.callback_manager = CallbackManager([self.callback_handler])

    def init_service_context(self):
        node_parser = SimpleNodeParser.from_defaults(
            include_metadata=False,
        )
        chatgpt = OpenAI(
            temperature=self.config.service_context.llm.llm_kwargs.temperature,
            model=self.config.service_context.llm.llm_kwargs.model_name,
        )
        service_context = ServiceContext.from_defaults(
            llm=chatgpt, node_parser=node_parser, callback_manager=self.callback_manager
        )
        self.service_context = service_context

    def init_response_synthesizer(self):
        response_synthesizer = get_response_synthesizer(
            response_mode=self.config.response_synthesizer.response_mode,
            use_async=self.config.response_synthesizer.use_async,
        )
        self.response_synthesizer = response_synthesizer

    def init_storage_context(self):
        storage_path = Path(
            self.config.storage_context.storage_path_root, self.config.index_id
        ).as_posix()  # TODO: figure out when/where to set storage_path
        self.storage_path = storage_path
        if not os.path.exists(self.storage_path):
            raise ValueError(
                f"Storage path {self.storage_path} does not exist. Create docstore.json first."
            )
        self.storage_context = StorageContext.from_defaults(
            persist_dir=self.storage_path
        )

    def load_summary_index(self):
        self.init_storage_context()
        self.index = load_index_from_storage(self.storage_context)

    def create_index(self):
        # Initialize ServiceContext and Response Synthesizer
        self.init_service_context()
        self.init_response_synthesizer()

        # Now proceed with creating the index
        self.index = DocumentSummaryIndex.from_documents(
            self.docs,
            include_metadata=False,
            summary_query=CUSTOM_SUMMARY_QUERY,
            response_synthesizer=self.response_synthesizer,
            service_context=self.service_context,
            show_progress=True,
        )

    def persist_index(self):
        storage_path = Path(
            self.config.storage_context.storage_path_root, self.config.index_id
        ).as_posix()
        self.storage_path = storage_path
        # self.init_storage_context()
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        self.index.storage_context.persist(storage_path)
