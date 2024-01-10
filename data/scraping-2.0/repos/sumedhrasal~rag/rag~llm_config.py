from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")


def get_embeddings(model='text-embedding-ada-002'):
    return OpenAIEmbeddings(model=model, openai_api_key=openai_api_key)


def get_model(model_name='text-ada-001'):
    return OpenAI(model_name=model_name, openai_api_key=openai_api_key, temperature=0.1)


def get_chat_model():
    return ChatOpenAI(openai_api_key=openai_api_key, temperature=0.5)

