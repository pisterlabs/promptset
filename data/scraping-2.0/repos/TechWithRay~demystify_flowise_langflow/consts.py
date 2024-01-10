import os
from dotenv import load_dotenv
import pinecone
from pathlib import Path
from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
)

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
SOURCE_DIRECTORY = f"{ROOT_DIR}/data"

# load environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")

# initialize pinecone
PINECONE_SETTINGS = {
    "api_key": PINECONE_API_KEY,          # find at app.pinecone.io
    "environment": PINECONE_ENVIRONMENT,  # next to api key in console
}

pinecone_db = pinecone.init(**PINECONE_SETTINGS)

# supported document types
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".xls": UnstructuredExcelLoader,
}
