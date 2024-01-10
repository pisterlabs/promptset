"""
All configs for the project are defined here.
"""
import openai
import os


openai.api_key = os.getenv("OPENAI_API_KEY", "")
USE_AZURE_AI = True
if USE_AZURE_AI:
    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_API_BASE", "")
    openai.api_version = os.getenv("AZURE_API_VERSION")

AZURE_GPT_ENGINE = "gpt35"

ENGINE_TOKENS_MAPPING = {
        "gpt35": 4096,  # gpt-35-turbo
        "gpt4-8k": 8192,  # gpt-4
        "gpt4-32k": 32768,  # gpt-4-32k
        "davinci": 4097  # text-davinci-003
    }
MAX_TOKENS = ENGINE_TOKENS_MAPPING.get(AZURE_GPT_ENGINE)

