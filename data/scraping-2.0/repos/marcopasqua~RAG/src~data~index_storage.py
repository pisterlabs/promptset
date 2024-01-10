import os
from contextlib import contextmanager
from multiprocessing import Lock
from typing import Tuple

from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.llms import OpenAI, HuggingFaceLLM, AzureOpenAI
from llama_index.indices.base import BaseIndex
from llama_index import (
    ServiceContext,
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex
)

from utils.logger import logger
from utils.constants import *



class IndexStorage:
    def __init__(self):
        self._current_model = "gpt-3.5-turbo"
        logger.info("initializing index and mongo ...")
        self._index = self.initialize_index()
        logger.info("initializing index and mongo done")
        self._lock = Lock()
        self._last_persist_time = 0

    @property
    def current_model(self):
        return self._current_model

    # def mongo(self):
    #     return self._mongo

    def index(self):
        return self._index

    @contextmanager
    def lock(self):
        # for the write operations on self._index
        with self._lock:
            yield

    def delete_doc(self, doc_id):
        """remove from both index and mongo"""
        with self.lock():
            pass
            # self._index.delete_ref_doc(doc_id, delete_from_docstore=True)
            # self._index.storage_context.persist(persist_dir=INDEX_PATH)
            # return self._mongo.delete_one({"doc_id": doc_id})

    def add_doc(self, nodes):
        """add to both index and mongo"""
        with self.lock():
            self._index.insert_nodes(nodes)
            self._index.storage_context.persist(persist_dir=INDEX_PATH)


    def initialize_index(self) -> BaseIndex:

        llm = AzureOpenAI(
            model="gpt-35-turbo-16k",
            deployment_name="gpt-35-turbo-16k",
            api_key=AZURE_API_KEY,
            azure_endpoint= AZURE_ENDPOINT,
            api_version= AZURE_API_VERSION,
        )


        embed_model = AzureOpenAIEmbedding(
            model="text-embedding-ada-002",
            deployment_name="text-embedding-ada-002",
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )

        # set your ServiceContext for all the next steps
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model
        )


        if os.path.exists(INDEX_PATH) and os.path.exists(os.path.join(INDEX_PATH,"docstore.json")):
            logger.info(f"Loading index from dir: {INDEX_PATH}")
            index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=INDEX_PATH),
                service_context=service_context,
            )
        else:
            index = VectorStoreIndex.from_documents([], service_context=service_context)
            index.storage_context.persist(persist_dir=INDEX_PATH)
        return index

# singleton of index storage
index_storage = IndexStorage()