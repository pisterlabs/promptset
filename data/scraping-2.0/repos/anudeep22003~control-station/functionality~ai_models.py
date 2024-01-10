# from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, OpenAIChat
import constants

llm_model_config = {
    "temperature": 0.7,
    "top_p": 1,
    "frequency_penalty": 0,
    "max_tokens": 256,
    "model_name": "text-davinci-003",
    "presence_penalty": 0,
    "streaming": True,
    "openai_api_key": str(constants.OPENAI_API_KEY),
}

chat_model_config = {
    # "cache": True,
    "openai_api_key": str(constants.OPENAI_API_KEY),
    "model_name": "gpt-3.5-turbo",
    "streaming": True,
}

# llm_client = OpenAI(**llm_model_config)
chat_client = OpenAIChat(**chat_model_config)
