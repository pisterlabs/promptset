from llama_index import StorageContext, SummaryIndex, VectorStoreIndex, download_loader, get_response_synthesizer, load_index_from_storage, load_indices_from_storage
from llama_index.node_parser import SimpleNodeParser
from store.virtualFileSystem.s3_service import S3Service
from store.vectorStore.fabrics.pinecone_vector_store_fabric import PineconeVectorStoreFabric
from llama_index.storage.docstore import MongoDocumentStore
from store.indexStore.index_store_manager import IndexStoreManager
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.indices.base import BaseIndex

from store.vectorStore.vector_store_manager import VectorStoreManager
import os


class IndexManager:

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance') or cls._instance is None:
            cls._instance = super(IndexManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.file_system_service = S3Service()

    def get_storage_context(self, namespace):
        index_store = IndexStoreManager.get_or_create()
        vector_store = PineconeVectorStoreFabric.get_vector_store(
            'awasome-index', namespace=namespace)

        mongo_uri = os.environ["MONGO_URI"]
        mongo_db_name = os.environ["MONGO_DB"]

        docstore = MongoDocumentStore.from_uri(
            uri=mongo_uri, db_name=mongo_db_name, namespace=namespace)

        storage_context = StorageContext.from_defaults(
            fs=self.file_system_service.fs, vector_store=vector_store, index_store=index_store, docstore=docstore)

        return storage_context

    async def upload_data_to_index_namespace(self, client_name, namespace):
        # client_path = self.file_system_service.get_path(client_name, namespace)

        documents = self.file_system_service.get_reader(
            f"{client_name}/{namespace}").load_data()

        storage_context = self.get_storage_context(namespace)

        for doc in documents:
            vector_index = VectorStoreIndex.from_documents(
                (doc,),
                storage_context=storage_context,
                show_progress=True,
            )

            # build summary index
            summary_index = SummaryIndex.from_documents(
                [doc],
                storage_context=storage_context,
                show_progress=True,
            )

            vector_index.set_index_id('vector_index:' + doc.get_doc_id())
            summary_index.set_index_id('summary_index:' + doc.get_doc_id())

    def get_index(self, namespace) -> BaseIndex:
        sc = self.get_storage_context(namespace)
        index = load_index_from_storage(sc, namespace)

        return index

    def get_indices(self, namespace) -> list[BaseIndex]:
        sc = self.get_storage_context(namespace)
        indices = load_indices_from_storage(sc)

        return indices

    def get_vector_index(self, namespace):
        sc = self.get_storage_context(namespace)
        index = VectorStoreIndex.from_vector_store(sc.vector_store, namespace)

        return index

    def ask_index(self, question: str, namespace: str):
        index = self.get_index(namespace)
        query = index.as_query_engine()

        return query.query(question)

    # def upload_local_data_to_index(self, dataPathFolder, l_index_name):
    #     vector_store = VectorStoreManager.create(l_index_name)
    #     index_store = IndexStoreManager.get_or_create()
    #     storage_context = StorageContext.from_defaults(
    #         index_store=index_store, vector_store=vector_store)

    #     required_exts = [".md"]
    #     documents = SimpleDirectoryReader(
    #         input_dir='/Users/mkucher/Documents/chat-bot/datacut/' + dataPathFolder, required_exts=required_exts, recursive=True
    #     ).load_data()

    #     doc_summary_index = GPTVectorStoreIndex.from_documents(
    #         documents,
    #         storage_context=storage_context,
    #         show_progress=True,
    #     )

    #     doc_summary_index.set_index_id(l_index_name)
