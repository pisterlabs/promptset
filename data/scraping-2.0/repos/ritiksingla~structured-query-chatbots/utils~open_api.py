import openai
import dotenv
import os
from langchain.llms import OpenAI

def get_openai_llm(temperature = 0, max_tokens = 700):
    dotenv.load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_type = os.getenv("OPENAI_API_TYPE") or "azure"
    openai.api_version = os.getenv("OPENAI_API_VERSION") or "2022-12-01"
    openai.api_base = os.getenv("OPENAI_API_BASE") or "https://oai-use-chatgpt-01.openai.azure.com"

    llm = OpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = temperature,
        max_tokens = max_tokens,
        streaming=True,
        model_kwargs= {
            "engine": "deploy-oai-use-chatgpt-01-text-davinci-003",
        }
    )
    return llm
