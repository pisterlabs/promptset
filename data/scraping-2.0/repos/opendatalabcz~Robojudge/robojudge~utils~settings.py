from pydantic import BaseSettings
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI


class Settings(BaseSettings):
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 4000
    SERVER_VERSION = "0.1.0"
    ENVIRONMENT = "dev"
    LOG_LEVEL = "INFO"

    CLIENT_HOST = "http://localhost"
    CLIENT_PORT = 3000

    EMBEDDING_DB_HOST = "chroma"
    EMBEDDING_DB_PORT = 8000
    EMBEDDING_CHUNK_SIZE = 500

    DOCUMENT_DB_HOST = "mongo"
    DOCUMENT_DB_PORT: int = 27017

    ENABLE_AUTOMATIC_SCRAPING: bool = True
    SCRAPER_MAX_RUN_CASE_COUNT = 30
    SCRAPER_TIMEOUT = 10_000
    SCRAPER_TASK_INTERVAL_IN_SECONDS = 3600
    OLDEST_KNOWN_CASE_ID = (
        450  # Based on manual testing, this is one of the first available cases
    )

    OPENAI_API_KEY = "dummykey"
    OPENAI_API_BASE = "https://api.openai.com/v1"
    OPENAI_API_TYPE = "azure"
    OPENAI_API_VERSION = "2023-08-01-preview"

    GPT_MODEL_NAME = "gpt-3.5-turbo-1106"
    BARD__Secure_1PSID = ""
    BARD__Secure_1PSIDTS = ""
    SUMMARIZE_MAX_PARALLEL_REQUESTS = 1
    DEFAULT_SUMMARIZE_LLM = "chatgpt"

    # Research answering
    AUTO_EVALUATOR_NAME = 'gpt-4-1106-preview'

    TOKENIZER_INPUT_LENGTH = 600
    TOKENIZER_MODEL = "czech-morfflex2.0-pdtc1.0-220710"
    TOKENIZER_URL = "http://lindat.mff.cuni.cz/services/morphodita/api/tag"

    AGENT_MAX_EXECUTION_TIME = 120

    REPLICATE_API_TOKEN: str = ''
    COHERE_API_TOKEN: str = ''

    class Config:
        env_file = ".env"


settings = Settings()