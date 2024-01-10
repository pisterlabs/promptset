import os

from dotenv import load_dotenv

from src.summarization.engines import LangChain, TensorFlow

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

CORS_CONFIG = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

OPEN_AI_TOKEN = os.getenv("OPEN_AI_TOKEN")
TENSORFLOW_MODEL_NAME = os.getenv("TENSORFLOW_MODEL_NAME")

# Carta automatica da engine a ser usada pelo service
LLM_ENGINES = {
    "LangChain": LangChain(api_key=OPEN_AI_TOKEN),
    "Tensorflow": TensorFlow(model_name=TENSORFLOW_MODEL_NAME),
}
