from os.path import realpath, join, dirname

from chromadb.config import Settings
from langchain.document_loaders.base import BaseLoader


ROOT_DIRECTORY = dirname(realpath(__file__))

SOURCE_DIRECTORY = join(ROOT_DIRECTORY, 'knowledge')

PERSIST_DIRECTORY = join(ROOT_DIRECTORY, 'db')

CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False
)
