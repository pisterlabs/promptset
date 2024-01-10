import os
import pickle
import pandas as pd
from app.configurations.development.config_parser import args
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.document_loaders.text import TextLoader
from app.configurations.development.settings import (
    VECTORSTORE_FILE,
    SCHEMA_FILE,
    AZURE_API_BASE,
    AZURE_API_KEY,
    AZURE_API_TYPE,
    MODEL_DEPLOYMENT_NAME,
    EMBED_DEPLOYMENT_NAME,
    EMBED_API_VERSION,
    MODEL_API_VERSION,
    OPENAI_API_KEY,
)


class DocumentRetrievalEngine:
    @staticmethod
    def _create_vectorstore_folder(folder_path):
        try:
            os.makedirs(folder_path)
            # TODO: Add this in verbose mode
            # print("The 'sessions' folder has been created.")
        except OSError as e:
            # TODO: Add this exception in logging
            # print(f"Error: Unable to create the 'sessions' folder - {e}")
            return

    @staticmethod
    def _load_pickle_file(pickle_file_path: str):
        """
        Loads the data from the pickle file.
        FIX ME: Write the return type.
        """
        # Load data from the pickle file
        with open(pickle_file_path, "rb") as file:
            data = pickle.load(file)

        return data

    @staticmethod
    def _save_vectorstore(vectorstore, file_path: str = VECTORSTORE_FILE) -> None:
        """
        Saves the vectorstore to a pickle file.
        """
        # Save vectorstore
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

    def _is_vectorstore_file_present(self, file_path: str = VECTORSTORE_FILE) -> bool:
        """
        Checks if the vectorstore file is present.
        """
        vectorstore_folder_path = os.path.dirname(file_path)

        if not os.path.exists(vectorstore_folder_path):
            self._create_vectorstore_folder(vectorstore_folder_path)
            return False

        return os.path.exists(file_path)

    @property
    def delete_vectorstore_file(self, file_path: str = VECTORSTORE_FILE) -> None:
        """
        Deletes the vectorstore file.
        """
        try:
            os.remove(file_path)
        except OSError as e:
            return

    def get_raw_data(self, file_path: str, data_type: str = "csv"):
        """
        FIX ME: Write the return type.
        """
        # Specify the path to the text file
        file_path = file_path

        if data_type == "csv":
            # Load the data from the csv file
            raw_data = pd.read_csv(file_path)
        elif data_type == "text":
            # Read the contents of the text file
            with open(file_path, "r") as file:
                raw_data = file.read()
        else:
            raise NotImplementedError

        return raw_data

    def get_raw_documents(self, file_path: str, loader_type: str = "csv"):
        """
        FIX ME: Write the return type.
        Later replace this with a function that gets data from the appropriate database.
        """
        raw_data = pd.read_csv(file_path)
        if loader_type == "csv":
            loader = CSVLoader(file_path=file_path)
        elif loader_type == "text":
            loader = TextLoader(file_path=file_path)
        else:
            raise NotImplementedError
        raw_documents = loader.load()

        return raw_documents

    def split_documents(
        self, raw_documents: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ):
        """
        FIX ME: Write the return type.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        documents = text_splitter.split_documents(raw_documents)
        return documents

    def create_vectorstore_data(self, documents):
        """
        FIX ME: Write the return type.
        """
        if args.personal_token:
            embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
            )
        else:
            embeddings = OpenAIEmbeddings(
                deployment=EMBED_DEPLOYMENT_NAME,
                openai_api_key=AZURE_API_KEY,
                openai_api_base=AZURE_API_BASE,
                openai_api_type=AZURE_API_TYPE,
                openai_api_version=EMBED_API_VERSION,
                chunk_size=16,
            )

        vectorstore_data = FAISS.from_documents(documents, embeddings)

        # Save vectorstore
        self._save_vectorstore(vectorstore_data)

        return vectorstore_data

    def get_vectorstore_data(
        self,
        schema_file_path: str = SCHEMA_FILE,
        vectorstore_file_path: str = VECTORSTORE_FILE,
    ):
        """
        FIX ME: Add code to check if vectorstore pickle file exists, if not then run ingest_data.py
        Write the return type.
        """
        if not self._is_vectorstore_file_present(vectorstore_file_path):
            raw_documents = self.get_raw_documents(schema_file_path)
            documents = self.split_documents(raw_documents)
            vectorstore_data = self.create_vectorstore_data(documents)
        else:
            vectorstore_data = self._load_pickle_file(vectorstore_file_path)
        return vectorstore_data
