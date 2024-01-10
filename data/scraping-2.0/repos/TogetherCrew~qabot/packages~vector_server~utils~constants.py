# Define the default values
import os

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

from logger.embedding_logger import logger

# from sentence_transformers import SentenceTransformer


ENV_FILE = os.getenv('ENV_FILE')

logger.debug(f"ENV_FILE: {ENV_FILE}")

if ENV_FILE != 'docker':
    dotenv_path = os.path.join(os.path.dirname(__file__), '../.local.env')
    logger.debug(f"Loading .env from {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"


#MongoDB
DB_CONNECTION_STR = os.getenv("DB_CONNECTION_STR", "")
assert DB_CONNECTION_STR, "DB_CONNECTION_STR environment variable is missing from .env"
DB_GUILD = os.getenv("DB_GUILD", "")
assert DB_GUILD, "DB_GUILD environment variable is missing from .env"

USE_LOCAL_STORAGE = True

DEEPLAKE_FOLDER = "vector_store"
DEEPLAKE_PLATFORM_FOLDER = "discord"

DEFAULT_EMBEDDINGS = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

DEEPLAKE_RAW_FOLDER = "DeepLake_VectorStore_414_419_raw_messages"
DEEPLAKE_SUMMARY_FOLDER = "DeepLake_VectorStore_414_419_summaries"

# VECTOR SERVER CONFIG
HIVEMIND_VS_PORT = os.getenv("HIVEMIND_VS_PORT", 1234)

# RABBITMQ CONFIG
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", 5672)
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")

# REDIS CONFIG
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = os.getenv('REDIS_PORT', 6379)
REDIS_USER = os.getenv('REDIS_USER', None)
REDIS_PASS = os.getenv('REDIS_PASS', None)

USER_AND_PASS = f"{REDIS_USER if REDIS_USER else '' }:{REDIS_PASS}@" if REDIS_PASS else ''

REDIS_URI = os.getenv('REDIS_URI', f"redis://{USER_AND_PASS}{REDIS_HOST}:{REDIS_PORT}")
