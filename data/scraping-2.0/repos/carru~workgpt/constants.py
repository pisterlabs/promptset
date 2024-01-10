import os
from chromadb.config import Settings
from langchain.document_loaders import UnstructuredPDFLoader

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/source_documents"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/db"

INGEST_THREADS = os.cpu_count() or 8

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

DOCUMENT_MAP = {
    ".pdf": (UnstructuredPDFLoader, {}),
}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# LLM - GPT4All
MODEL_PATH='models/ggml-gpt4all-j-v1.3-groovy.bin'

# Confluence
CONFLUENCE_TOKEN = os.environ.get('CONFLUENCE_TOKEN')
CONFLUENCE_BASE_URL = os.environ.get('CONFLUENCE_BASE_URL')
CONFLUENCE_SPACES_TO_LOAD = os.environ.get('CONFLUENCE_SPACES_TO_LOAD', '').split(',')
CONFLUENCE_MAX_PAGES_PER_SPACE = int(os.environ.get('CONFLUENCE_MAX_PAGES_PER_SPACE', '1000'))
CONFLUENCE_CHUNCK_SIZE = int(os.environ.get('CONFLUENCE_CHUNCK_SIZE', '1000'))
CONFLUENCE_CHUNCK_OVERLAP = int(os.environ.get('CONFLUENCE_CHUNCK_OVERLAP', '100'))
CONFLUENCE_RETRIES = int(os.environ.get('CONFLUENCE_RETRIES', '3'))
CONFLUENCE_MIN_RETRY_S = int(os.environ.get('CONFLUENCE_MIN_RETRY_S', '2'))
CONFLUENCE_MAX_RETRY_S = int(os.environ.get('CONFLUENCE_MAX_RETRY_S', '10'))

# WebUI
WEBUI_CONFLUENCE_SPACES_READ = os.environ.get('WEBUI_CONFLUENCE_SPACES_READ')
WEBUI_APP_NAME = os.environ.get('WEBUI_APP_NAME', 'WorkGPT')