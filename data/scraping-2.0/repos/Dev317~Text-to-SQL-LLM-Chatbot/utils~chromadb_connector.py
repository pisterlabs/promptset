import streamlit as st
from streamlit.connections import ExperimentalBaseConnection
import chromadb
from chromadb.utils.embedding_functions import *
import pandas as pd
import uuid
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from typing import Dict
from typing_extensions import override
import tempfile


class ChromaDBConnection(ExperimentalBaseConnection):
    """
    This class acts as an adapter to connect to ChromaDB vector database.
    It extends the ExperimentalBaseConnection class by overidding _connect(),
    and also provides other helpful methods to interact with the ChromaDB.
    """

    @override
    def _connect(self, **kwargs) -> chromadb.Client:
        type = self._kwargs["client_type"]

        if type == "PersistentClient":
            path = self._kwargs["path"] if "path" in self._kwargs else f"{tempfile.gettempdir()}/chroma"

            return chromadb.PersistentClient(
                path=path,
            )

        if type == "HttpClient":
            return chromadb.HttpClient(
                host=self._kwargs["host"],
                port=self._kwargs["port"],
                ssl=self._kwargs["ssl"],
            )

        return chromadb.Client()

    def create_collection(self,
                          collection_name: str,
                          embedding_function_name: str,
                          config: Dict) -> None:

        embedding_function = DefaultEmbeddingFunction()
        if embedding_function_name == "VertexEmbedding":
            embedding_function = GoogleVertexEmbeddingFunction(**config)
        elif embedding_function_name == "OpenAIEmbedding":
            embedding_function = OpenAIEmbeddingFunction(**config)
        try:
            self._raw_instance.create_collection(name=collection_name,
                                                 embedding_function=embedding_function)
        except Exception as ex:
            raise ex

    def delete_collection(self, collection_name: str) -> None:
        try:
            self._raw_instance.delete_collection(name=collection_name)
        except Exception as ex:
            raise ex

    def get_collection_names(self) -> List:
        collection_names = []
        collections = self._raw_instance.list_collections()
        for col in collections:
            collection_names.append(col.name)
        return collection_names

    def get_collection_data(self,
                            collection_name: str,
                            attributes: List = ["documents", "embeddings", "metadatas"]):

        @st.cache_data(ttl=10)
        def get_data():
            collection = self._raw_instance.get_collection(collection_name)
            collection_data = collection.get(
                include=attributes
            )
            return pd.DataFrame(data=collection_data)
        return get_data()

    def get_collection_embedding_function(self, collection_name: str):
        collection = self._raw_instance.get_collection(collection_name)
        return collection._embedding_function.__class__.__name__

    def retrieve(self,
                 collection_name: str,
                 query: str) -> pd.DataFrame:
        collection = self._raw_instance.get_collection(collection_name)
        embeddings = collection._embedding_function.__call__(query)
        results = collection.query(
            query_embeddings=embeddings,
            n_results=10,
            include=["documents", "distances", "embeddings"]
        )
        df = pd.DataFrame(data=results)
        return df[["ids", "distances", "embeddings", "documents"]]


    def upload_document(self,
                        directory: str,
                        collection_name: str,
                        file_paths: List) -> None:
        collection = self._raw_instance.get_collection(collection_name)

        try:
            loader = DirectoryLoader(directory, glob="*.*")
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)

            for doc in docs:
                fid = f"{str(uuid.uuid4())}"
                embedding = collection._embedding_function([doc.page_content])
                source = doc.metadata['source'].split("/")[-1]
                collection.add(ids=[fid],
                            metadatas={'source': source},
                            documents=doc.page_content,
                            embeddings=embedding)

            for file_path in file_paths:
                os.remove(file_path)
        except Exception as ex:
            raise ex