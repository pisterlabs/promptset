import os

from chromadb.config import Settings
from huggingface_hub import hf_hub_download

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredHTMLLoader

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".html": UnstructuredHTMLLoader,
}

# Select the Model ID and model_basename.
# You also have to set various parameter for this model.
# If you choose a non Intructor embedding model, you also have to change the HuggingFaceInstructEmbeddings import
# in both ingest.py and run_localGPT.py

# # 13B: high quality answer, for higher-end computer, with 16GB of RAM and GPU recommanded.
# EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"
# EMBEDDING_MODEL_PATH = None # Set to None to download the model from HuggingFace
# MODEL_ID = "TheBloke/OpenOrca-Platypus2-13B-GGML"
# MODEL_NAME = "openorca-platypus2-13b.ggmlv3.q4_K_M.bin"
# MODEL_PATH = None # Set path the the model, if None it will download the model from HuggingFace
# MODEL_MAX_CTX_SIZE = 4096
# MODEL_STOP_SEQUENCE = ["###"]
# MODEL_GPU_LAYERS = 25 # offload part of the model to GPU, improving slightly performance. If you have enought VRAM, you can offload the entire model for a significant speed boost. 25 layers should fit in 8GB of VRAM
# MODEL_TEMPERATURE = 0.4
# MODEL_PREFIX = {"human": "Question", "ai": "Response"}
# MODEL_PROMPT_TEMPLATE = """### Instruction:
# Use the following pieces of Context to answer the Question at the end. Take into acount the conversation History.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context:
# {context}

# Conversation History:
# {history}

# Question:
# {question}

# ### Response:
# """

# 7B: good answer, for high-end computer, with 16GB of available RAM.
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en"
MODEL_ID = "TheBloke/orca_mini_v3_7B-GGML"
MODEL_NAME = "orca_mini_v3_7b.ggmlv3.q4_K_M.bin"
MODEL_PATH = None # Set path the the model, if None it will download the model from HuggingFace
MODEL_MAX_CTX_SIZE = 4096
MODEL_STOP_SEQUENCE = ["###"]
MODEL_GPU_LAYERS = 0
MODEL_TEMPERATURE = 0.4
MODEL_PREFIX = {"human": "User", "ai": "Assistant"}
MODEL_PROMPT_TEMPLATE = """### System:
Use the following pieces of Context to answer the the User at the end. Take into acount the conversation History.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Conversation History:
{history}

### User:
{question}

### Assistant:
"""

if (MODEL_PATH is None) or (not os.path.isfile(MODEL_PATH)):
    MODEL_PATH = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_NAME, local_dir="models", cache_dir="models/.cache")
