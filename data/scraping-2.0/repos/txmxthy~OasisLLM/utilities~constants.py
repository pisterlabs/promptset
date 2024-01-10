import os

from dotenv import load_dotenv
from chromadb.config import Settings

from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
)

load_dotenv()
UTIL_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(UTIL_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
INGESTION_DIR = os.path.join(DATA_DIR, "ingestion")
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma")

# Define the Chroma settings
CHROMA_CFG = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=CHROMA_PERSIST_DIR,
    anonymized_telemetry=False
)

# @TODO - list supported and add more document types
# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCTYPE_LOADERS = {
    ".txt": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader
}
