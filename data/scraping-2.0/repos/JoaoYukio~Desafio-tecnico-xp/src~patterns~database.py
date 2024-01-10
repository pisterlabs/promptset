from abc import ABC, abstractmethod

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.docstore.document import Document

from langchain.document_loaders import DirectoryLoader
import os
import shutil

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt


class DatabaseInterface(ABC):
    @abstractmethod
    def connect_fromDoc(self, docs: list):
        pass

    @abstractmethod
    def connect_fromText(self, text: str):
        pass

    @abstractmethod
    def query(self, query_string: str):
        pass

    @abstractmethod
    def load_documents(self, directory: str) -> Document:
        pass

    @abstractmethod
    def doc_splitter(self, documents: str, chunk_size: int, chunk_overlap: int) -> list:
        pass

    @abstractmethod
    def text_splitter(self, text: str, chunk_size: int, chunk_overlap: int) -> list:
        pass

    @abstractmethod
    def append_documents(self, documents: list):
        pass

    @abstractmethod
    def append_text(self, text: str):
        pass


class ChromaDatabase(DatabaseInterface):
    def __init__(self, embeddings: OpenAIEmbeddings, persist_directory: str):
        self.embeddings = embeddings
        self.persist_directory = persist_directory

        def is_directory_empty(dir_path):
            return not bool(os.listdir(dir_path))

        if is_directory_empty(self.persist_directory):
            print("DB vazio")
            self.docsearch = None
        else:
            self.docsearch = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            print("Database loaded successfully.")

    def load_documents(self, directory: str) -> Document:
        return DirectoryLoader(directory).load()

    def load_text(self, directory: str) -> str:
        return DirectoryLoader(directory).load_text()

    def doc_splitter(self, documents: str, chunk_size=1000, chunk_overlap=20) -> list:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ).split_documents(documents)

    def text_splitter(self, text: str, chunk_size=1000, chunk_overlap=20) -> list:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ).split_text(text)

    def connect_fromDoc(self, docs: list):
        # if not self.docsearch:
        #     self.docsearch = Chroma.from_documents(
        #         documents=self.text_splitter(docs),
        #         embedding=self.embeddings,
        #         persist_directory=self.persist_directory,
        #     )
        #     self.docsearch.persist()
        # else:
        #     print("Erro: Banco de dados Chroma já está conectado.")
        self.docsearch = Chroma.from_documents(
            documents=self.doc_splitter(docs),
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.docsearch.persist()

    def connect_fromText(self, text: str):
        # if not self.docsearch:
        #     print("Criando banco de dados Chroma...")
        #     self.docsearch = Chroma.from_texts(
        #         texts=self.text_splitter(text),
        #         embedding=self.embeddings,
        #         persist_directory=self.persist_directory,
        #     )
        #     self.docsearch.persist()
        # else:
        #     print("Erro: Banco de dados Chroma já está conectado.")
        self.docsearch = Chroma.from_texts(
            texts=self.text_splitter(text),
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.docsearch.persist()

    def append_documents(self, documents: list) -> None:
        if not self.docsearch:
            print("Erro: Banco de dados Chroma não está conectado.")
            return
        self.docsearch.add_documents(documents=documents)

    def append_text(self, text: str) -> None:
        if not self.docsearch:
            print("Erro: Banco de dados Chroma não está conectado.")
            return
        self.docsearch.add_texts(texts=text)

    def query(self, query_string: str, num_res=5) -> list:
        if self.docsearch:
            return self.docsearch.similarity_search(query_string, k=num_res)
        else:
            print("Erro: Banco de dados Chroma não está conectado.")
            return []

    def get_vector_store(self):
        return self.docsearch

    def get_vectors(self) -> list:
        vectors = self.docsearch.get()
        return vectors

    def delete_persistent_database(self) -> str:
        try:
            if self.docsearch:
                self.docsearch.delete_collection()
                self.docsearch = None
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                return f"Banco de dados em {self.persist_directory} foi excluído com sucesso."
            else:
                return f"Nenhum banco de dados encontrado em {self.persist_directory}."
        except PermissionError:
            return f"Erro: Não foi possível excluir o banco de dados em {self.persist_directory} porque está sendo usado por outro processo."


class DatabaseFactory:
    @staticmethod
    def create_database(database_type: str, **kwargs) -> DatabaseInterface:
        """
        Cria uma instância de banco de dados com base no tipo fornecido.

        :param database_type: Tipo do banco de dados ('chroma', 'pinecone', etc.)
        :param kwargs: Argumentos adicionais necessários para inicializar o banco de dados.
        :return: Uma instância do banco de dados.
        """
        if database_type == "chroma":
            embeddings = kwargs.get("embeddings", "")
            persist_directory = kwargs.get("persist_directory", "./data/chroma_store")
            return ChromaDatabase(
                embeddings=embeddings, persist_directory=persist_directory
            )

        # elif database_type == 'pinecone':
        #     return PineconeDatabase(**kwargs)

        else:
            raise ValueError(f"Tipo de banco de dados '{database_type}' não suportado.")
