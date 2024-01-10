# Define the default values
import os

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

from logger.hivemind_logger import logger

ENV_FILE = os.getenv('ENV_FILE')

logger.debug(f"ENV_FILE: {ENV_FILE}")

if ENV_FILE != 'docker':
    dotenv_path = os.path.join(os.path.dirname(__file__), '../../.local.env')
    logger.debug(f"Loading .env from {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

# Set Agent Settings
AGENT_NAME = os.getenv("AGENT_NAME", "")
assert AGENT_NAME, "AGENT_NAME variable is missing from .env"
AGENT_ROLE = os.getenv("AGENT_ROLE", "")
assert AGENT_ROLE, "AGENT_ROLE variable is missing from .env"
AGENT_OBJECTIVE = os.getenv("AGENT_OBJECTIVE", None)

# API CONFIG
HIVEMIND_API_PORT = os.getenv("HIVEMIND_API_PORT", 3333)

# VECTOR SERVER CONFIG
HIVEMIND_VS_HOST = os.getenv("HIVEMIND_VS_HOST", "http://localhost")
HIVEMIND_VS_PORT = os.getenv("HIVEMIND_VS_PORT", 1234)

VECTOR_SERVER_URL = f"{HIVEMIND_VS_HOST}:{HIVEMIND_VS_PORT}"

# RABBITMQ CONFIG
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", 5672)
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")

DEFAULT_AGENT_DIR = os.path.join(os.path.dirname(__file__), "../agent_data")

DEFAULT_EMBEDDINGS = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Define the base path for the serialization
BASE_PATH_SERIALIZATION = os.path.join(DEFAULT_AGENT_DIR, "serialization")