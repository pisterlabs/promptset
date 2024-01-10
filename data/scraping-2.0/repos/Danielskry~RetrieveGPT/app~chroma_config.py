''' Chroma config '''
from chromadb.config import Settings

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from app.utils.load_yaml_config import load_yaml_config

# Define document loaders mapping
DOC_LOADERS_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

# Load Chroma configuration from YAML file
chroma_config = load_yaml_config("CHROMA_CONFIGURATION_PATH")

if chroma_config is None:
    raise ValueError("Failed to load embeddings configuration.")

# Create Chroma settings object
CHROMA_SETTINGS = Settings(
    persist_directory=chroma_config['chroma_config']['persist_directory_path'],
    chroma_db_impl=chroma_config['chroma_config']['chroma_db_impl'],
    anonymized_telemetry=chroma_config['chroma_config']['anonymized_telemetry'],
)
