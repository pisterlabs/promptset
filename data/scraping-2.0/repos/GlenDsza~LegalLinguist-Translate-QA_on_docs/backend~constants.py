# CONSTANTS

import os
from chromadb.config import Settings
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

SOURCE_DIRECTORY = "../input/newdata/data"
PERSIST_DIRECTORY = ""

MODELS_PATH = "./models"

INGEST_THREADS = 2
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# Context Window and Max New Tokens
CONTEXT_WINDOW_SIZE = 2048
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

N_GPU_LAYERS = 60
N_BATCH = 512

DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    # ".pdf": PDFMinerLoader,
    ".pdf": UnstructuredFileLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Uses 1.5 GB of VRAM (High Accuracy with lower VRAM usage)
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
# EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl" # Uses 5 GB of VRAM (Most Accurate of all models)


MODEL_ID = "TheBloke/zephyr-7B-alpha-GGUF"
MODEL_BASENAME = "zephyr-7b-alpha.Q8_0.gguf"

device_type = "cuda"
model_type = "mistral"

# PROMPT TEMPLATES


system_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user other than provided context."""


def get_prompt_template(system_prompt, history=False):
    B_INST, E_INST = "<s>[INST] ", " [/INST]"
    if history:
        prompt_template = (
            B_INST
            + system_prompt
            + """

        Context: {history} \n {context}
        User: {question}"""
            + E_INST
        )
        prompt = PromptTemplate(
            input_variables=["history", "context", "question"], template=prompt_template)
    else:
        prompt_template = (
            B_INST
            + system_prompt
            + """

        Context: {context}
        User: {question}"""
            + E_INST
        )
        prompt = PromptTemplate(
            input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferMemory(
        input_key="question", memory_key="history")

    return (
        prompt,
        memory,
    )
