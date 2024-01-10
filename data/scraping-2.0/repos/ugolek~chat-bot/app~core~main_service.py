from http import HTTPStatus
from fastapi import HTTPException
from core.retriever_service import RetrieverService
from store.virtualFileSystem.s3_service import S3Service
from core.indexManager.index_manager import IndexManager
from core.service_context_service import init_global_service_context
from langchain.chat_models import ChatOpenAI
from llama_index import LLMPredictor, LangchainEmbedding, ServiceContext, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores import PineconeVectorStore, ChromaVectorStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.llms import OpenAI
from llama_index.storage.index_store import MongoIndexStore
from llama_index.node_parser import SentenceWindowNodeParser
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine
import pinecone
import os


class MainService:
    def __init__(self):
        self.index_manager = IndexManager()
        self.docstore_service = S3Service()
        self.retriever_service = RetrieverService()
        init_global_service_context()

    # def upload_local_data(self, dataPathFolder, l_index_name):
    #     return self.index_manager.upload_data_to_index(dataPathFolder, l_index_name)

    async def upload_files_to_namespace(self, files, client_name: str, namespace: str, path: str = ''):
        await self.docstore_service.upload_files(files, client_name, namespace, path)

        return await self.index_manager.upload_data_to_index_namespace(client_name, namespace)

    def ask_namespace(self, question: str, namespace: str):
        return self.index_manager.ask_index(question, namespace)

    def ask_namespace_with_recursive_retriever(self, question: str, namespace: str):
        indices = self.index_manager.get_indices(namespace)

        storage_context = self.index_manager.get_storage_context(namespace)

        retriever = self.retriever_service.get_recursive_retriever(indices, storage_context)

        response_synthesizer = get_response_synthesizer()

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        response = query_engine.query(question)

        return response
