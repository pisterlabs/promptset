import sys

sys.path.insert(0, ".")
sys.path.insert(0, "..")

import os
import openai
from langchain.document_loaders.csv_loader import CSVLoader
from src.config import *

from langchain.document_loaders import (
    PyPDFLoader,
    EverNoteLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    BSHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    Docx2txtLoader,
    UnstructuredXMLLoader,
    JSONLoader,
    # S3DirectoryLoader,
    # S3FileLoader,
    # AzureBlobStorageContainerLoader,
    # GCSDirectoryLoader,
    # GCSFileLoader,
    # GoogleDriveLoader,
    # HuggingFaceDatasetLoader,
    # NotionDirectoryLoader,
    # NotionDBLoader,
    # SlackDirectoryLoader,
    # DirectoryLoader,
)

from src.config import (
    CHROMA_SETTINGS,
    PERSIST_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    DEVICE_TYPE,
)

EXTENSION_MAPPING = {
    ".txt": TextLoader,
    ".py": TextLoader,
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlxs": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".xml": UnstructuredXMLLoader,
    ".json": JSONLoader,
    ".md": UnstructuredMarkdownLoader,
}

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import tempfile


def ingest(file_path):
    # load documents
    # loader = PyPDFLoader(file)
    loader = TextLoader(file_path)
    # loader = load_single_document(file_path)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE_TYPE},
        encode_kwargs={"normalize_embeddings": True},
    )
    # create vector database from data
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    # st.write(f"db persist path: {PERSIST_DIRECTORY}")
    db.persist()
    return db
