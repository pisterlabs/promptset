import os

from chromadb.config import Settings

from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).parent

SOURCE_DIRECTORY = f"{ROOT_DIR}/data/source"

PERSIST_DIRECTORY = f"{ROOT_DIR}/data/db"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)

DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".xls": UnstructuredExcelLoader,
}
