
from pydantic import BaseSettings
class Settings(BaseSettings):
    AOAI_API_KEY :str
    AOAI_ENDPOINT :str
    AOAI_MODEL :str

settings = Settings(_env_file=".env")

from logger import logger

from openai import AzureOpenAI

def call_chatgpt (messages,temperature=0):

    client = AzureOpenAI(
        azure_endpoint = settings.AOAI_ENDPOINT, 
        api_key=settings.AOAI_API_KEY,  
        api_version="2023-07-01-preview",
        timeout=20,
        max_retries=3,
    )

    response = client.chat.completions.create(
    model=settings.AOAI_MODEL, # model = "deployment_name".
    messages=messages,
    temperature=temperature,
    )

    return response.choices[0].message.content


def call_chatgpt_w_token (messages,temperature=0):

    client = AzureOpenAI(
        azure_endpoint = settings.AOAI_ENDPOINT, 
        api_key=settings.AOAI_API_KEY,  
        api_version="2023-07-01-preview",
        timeout=20,
        max_retries=3,
    )

    response = client.chat.completions.create(
    model=settings.AOAI_MODEL, # model = "deployment_name".
    messages=messages,
    temperature=temperature,
    )

    return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens


def call_chatgpt_w_function(messages,functions,temperature=0):

    client = AzureOpenAI(
        azure_endpoint = settings.AOAI_ENDPOINT, 
        api_key=settings.AOAI_API_KEY,  
        api_version="2023-10-01-preview",
        timeout=10,
        max_retries=3,
    )

    response = client.chat.completions.create(
        model=settings.AOAI_MODEL,
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    return response.choices[0].message
