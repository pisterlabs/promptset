from langchain.embeddings.openai import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from langchain.chat_models import AzureChatOpenAI
from llama_index.llms import AzureOpenAI
import os
from dotenv import load_dotenv
import openai
load_dotenv()


AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_BASE = os.getenv("AZURE_OPENAI_BASE")
AZURE_OPENAI_TYPE = os.getenv("AZURE_OPENAI_TYPE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_BASE
openai.api_version = AZURE_OPENAI_API_VERSION
openai.api_key = AZURE_OPENAI_KEY

AZURE_OPENAI_CHAT_AGENT_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_CHAT_AGENT_DEPLOYMENT_NAME")
AZURE_OPENAI_CHAT_AGENT_MODEL_NAME = os.getenv(
    "AZURE_OPENAI_CHAT_AGENT_MODEL_NAME")

AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_VERSION = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_VERSION")
AZURE_OPENAI_EMBEDDING_MODEL_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDING_MODEL_NAME")

EMBEDDING_LLM = LangchainEmbedding(OpenAIEmbeddings(
    deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    model=AZURE_OPENAI_EMBEDDING_MODEL_NAME,
    openai_api_type=AZURE_OPENAI_TYPE,
    openai_api_key=AZURE_OPENAI_KEY,
    openai_api_base=AZURE_OPENAI_BASE,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    chunk_size=1
))

EMBEDDING_ENGINE = OpenAIEmbeddings(
    deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    model=AZURE_OPENAI_EMBEDDING_MODEL_NAME,
    openai_api_type=AZURE_OPENAI_TYPE,
    openai_api_key=AZURE_OPENAI_KEY,
    openai_api_base=AZURE_OPENAI_BASE,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    chunk_size=1
)


AGENT_LLM = AzureChatOpenAI(
    openai_api_type=AZURE_OPENAI_TYPE,
    openai_api_key=AZURE_OPENAI_KEY,
    openai_api_base=AZURE_OPENAI_BASE,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_CHAT_AGENT_DEPLOYMENT_NAME,
    model_name=AZURE_OPENAI_CHAT_AGENT_MODEL_NAME,
    temperature=0
)
