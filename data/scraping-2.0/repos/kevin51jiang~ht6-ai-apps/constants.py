import os
from dotenv import load_dotenv, find_dotenv

from chromadb.config import Settings
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Choose loaders based on the type of document you want to use
# https://python.langchain.com/docs/integrations/document_loaders/
from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)

# Load the .env file
load_dotenv(find_dotenv())

# Grab our API key
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")


# Set current directory
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Can be changed to a specific number (only affects speed in parallel mode)
INGEST_THREADS = os.cpu_count() or 8

LOCAL_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# LOCAL_EMBEDDING_MODEL_NAME = "hkunlp/instructor-large" # More powerful

# choose which type of text loader to use for each file
# e.g. TextLoader might just grab the text from inside the text document, but YoutubeLoader might look at the transcripts
# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
}
