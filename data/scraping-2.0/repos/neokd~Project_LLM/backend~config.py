import os
import torch
from chromadb.config import Settings
# from langchain.document_loaders import (
#     CSVLoader,
#     TextLoader,
#     UnstructuredExcelLoader,
#     Docx2txtLoader,
# )
# from langchain.document_loaders import (
#     UnstructuredFileLoader,
#     UnstructuredMarkdownLoader,
# )

from langchain.document_loaders import (
    CSVLoader,
    JSONLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredTSVLoader,
    Docx2txtLoader,
    PDFMinerLoader
)

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

DOCUMENT_MAP = {
    ".pdf": PDFMinerLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".html": UnstructuredHTMLLoader,
    ".tsv": UnstructuredTSVLoader,
    ".epub": UnstructuredEPubLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
    ".json": JSONLoader,
}


if torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
else:
    DEVICE_TYPE = "cpu"


SOURCE_DIRECTORY = os.path.join(os.path.dirname(__file__), "source")
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "db")
STRUCTURE_DIRECTORY = os.path.join(os.path.dirname(__file__), "structure.json")


# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

CONTEXT_WINDOW_SIZE = 4096

MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE 
N_GPU_LAYERS = 40

DEFAULT_MEMORY_KEY = 2


N_BATCH = 512

# Default Instructor Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
MODEL_NAME = os.getenv("MODEL_NAME", "TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
MODEL_FILE = os.getenv("MODEL_FILE", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")
