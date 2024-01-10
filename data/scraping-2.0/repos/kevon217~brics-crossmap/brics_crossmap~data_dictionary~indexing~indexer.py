from pathlib import Path
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    LangchainEmbedding,
    set_global_service_context,
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores.utils import node_to_metadata_dict
from llama_index.schema import MetadataMode
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import chromadb
from chromadb.utils import embedding_functions

from brics_crossmap.data_dictionary.indexing.utils import batchify


class Indexer:
    def __init__(self, cfg, client):
        self.cfg = cfg
        self.client = client
        self.service_context = self.initialize_service_context()

    def initialize_service_context(self):
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(
                model_name=self.cfg.indices.index.collections.embed.model_name,
                model_kwargs={},
                encode_kwargs=self.cfg.indices.index.collections.embed.model_kwargs,
            ),
            embed_batch_size=self.cfg.indices.index.collections.embed.model_kwargs.batch_size,
        )
        service_context = ServiceContext.from_defaults(
            embed_model=embed_model, llm=None
        )
        set_global_service_context(service_context)
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.cfg.indices.index.collections.embed.model_name
            )
        )
        return service_context

    def add_nodes(self, nodes, collection_name):
        collection = self.client.get_or_create_collection(
            collection_name,
            metadata=dict(self.cfg.indices.index.collections.distance_metric),
            embedding_function=self.embedding_function,
        )

        vector_store = ChromaVectorStore(collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Index nodes in batches
        for batch in batchify(nodes, self.cfg.indices.index.collections.max_batch_size):
            index = VectorStoreIndex(
                batch,
                storage_context=storage_context,
                service_context=self.service_context,
                show_progress=True,
            )

            # Persist each batch
            storage_path_index = Path(
                self.cfg.indices.index.storage_path_root
            ).as_posix()
            index.storage_context.persist(storage_path_index)

    def update_nodes(self, nodes, collection_name):
        collection = self.client.get_or_create_collection(
            collection_name,
            metadata=dict(self.cfg.indices.index.collections.distance_metric),
            embedding_function=self.embedding_function,
        )

        metadatas = []
        ids = []
        documents = []
        for n in nodes:
            id_ = n.id_
            metadata = node_to_metadata_dict(n, remove_text=True, flat_metadata=True)
            document = n.get_content(metadata_mode=MetadataMode.NONE)
            ids.append(id_)
            metadatas.append(metadata)
            documents.append(document)
        collection.upsert(
            ids=ids,
            metadatas=metadatas,
            documents=documents,
        )
