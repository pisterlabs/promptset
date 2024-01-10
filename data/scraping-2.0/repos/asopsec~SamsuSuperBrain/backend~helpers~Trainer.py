import os
import pathlib
import pickle
import shutil

from config import apikeys

from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader, Docx2txtLoader, TextLoader, \
    UnstructuredMarkdownLoader, CSVLoader, JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = apikeys.OPENAI_API_KEY

class Trainer:

    def is_directory_empty(self, directory_path):
        # Check if the directory exists
        print('dp:', directory_path)
        if not os.path.isdir(directory_path):
            raise ValueError("Invalid directory path")

        # Get the list of files and directories in the specified directory
        items = os.listdir(directory_path)

        # Filter out directories from the list
        files = [item for item in items if os.path.isfile(os.path.join(directory_path, item))]

        # Return True if the directory has any files, False otherwise
        return bool(files)

    @staticmethod
    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata["question"] = record.get("question")

        for index, question_variant in enumerate(record.get("question_variants")):
            metadata[f"question_variant_{index+1}"] = question_variant

        return metadata

    def get_documents(self) -> list:
        documents = []
        documents_path = pathlib.Path(__file__).parent.parent.parent.absolute()

        documents_path = documents_path.joinpath('training', 'datasets')

        if not self.is_directory_empty(documents_path):
            return documents

        for document_path in documents_path.iterdir():
            if document_path.is_file():
                documents.append(document_path)

        return documents

    def delete_documents(self, document_paths: list):
        # Delete Documents from datasets/{language}/{brand}
        for document_path in document_paths:
            if document_path.is_file():
                document_path.unlink()

    def get_vectorstore(self) -> list:
        # Get Vectorstore from vectorstores/vectorstore.pkl
        vectorstore = None
        vectorstore_path = pathlib.Path(__file__).parent.parent.parent.absolute()
        vectorstore_path = vectorstore_path.joinpath("vectorstores", "vectorstore.pkl")
        if vectorstore_path.is_file():
            with open(vectorstore_path, "rb") as f:
                vectorstore = pickle.load(f)
        return vectorstore

    def train(self):
        documents = []

        documents_path = pathlib.Path(__file__).parent.parent.parent.absolute()

        documents_path = documents_path.joinpath('training', 'datasets')

        for file in os.listdir(documents_path):
            if file.endswith('.pdf'):
                pdf_path = str(documents_path.joinpath(file))
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            elif file.endswith('.json'):
                json_path = str(documents_path.joinpath(file))
                loader = JSONLoader(
                    file_path=json_path,
                    jq_schema='.[]',
                    content_key="answer",
                    metadata_func=self.metadata_func
                )
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                doc_path = str(documents_path.joinpath(file))
                loader = Docx2txtLoader(doc_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                text_path = str(documents_path.joinpath(file))
                loader = TextLoader(text_path)
                documents.extend(loader.load())
            elif file.endswith('.md'):
                markdown_path = str(documents_path.joinpath(file))
                loader = UnstructuredMarkdownLoader(markdown_path)
                documents.extend(loader.load())
            elif file.endswith('.csv'):
                csv_path = str(documents_path.joinpath(file))
                loader = CSVLoader(csv_path)
                documents.extend(loader.load())

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        chunked_documents = text_splitter.split_documents(documents)

        # Embed and store the texts
        # Supplying a persist_directory will store the embeddings on disk

        persist_directory = f'training/vectorstores/'

        # Remove old vectorstore
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        # Create directory if not exists
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        # here we are using OpenAI embeddings but in future we will swap out to local embeddings
        embedding = OpenAIEmbeddings()

        vectordb = Chroma.from_documents(documents=chunked_documents,
                                         embedding=embedding,
                                         persist_directory=persist_directory)

        # persist the db to disk
        vectordb.persist()
        # self.delete_documents(document_paths)

        return 'Training complete'

if __name__ == '__main__':
    pass