# config.py
# import logging
from pathlib import Path

from langchain.chat_models import ChatOpenAI

# Development Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
LOGS_DIR = Path(BASE_DIR, "logs")

# Data Directories
DATA_DIR = Path("/data/DATASCI")
RAW_DATA = Path(DATA_DIR, "raw")
INTERMEDIATE_DIR = Path(DATA_DIR, "intermediate")
RESULTS_DIR = Path(DATA_DIR, "results")

# DATA Files

# Assets
ASSETS_DIR = Path(BASE_DIR, "assets")
TEMPLATE = Path(ASSETS_DIR, "numbered_template.docx")

CHAT = ChatOpenAI(
    openai_api_base="https://mockgpt.wiremockapi.cloud/v1",
    openai_api_key="sk-aqrgjxkpilpc1wlpjeg0gfsc9zxjh3zr",
    model="gpt-3.5-turbo"
)

PUBMED_CHAT = ChatOpenAI(
    openai_api_base="https://mockgpt.wiremockapi.cloud/v1",
    openai_api_key="sk-aqrgjxkpilpc1wlpjeg0gfsc9zxjh3zr",
    model="gpt-4"
)


# lit search
MAX_PUBMED_RESULTS = 50
MIN_ARTICLES = 10

