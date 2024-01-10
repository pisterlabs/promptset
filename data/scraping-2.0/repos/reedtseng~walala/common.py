import os
import openai

from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI

openai.api_type = "azure"

chat_deploy = "gpt-35-turbo-16k"
chat_model = "gpt-3.5-turbo-16k"
openai_api_version = "2023-07-01-preview"

llm = AzureOpenAI(
    deployment_name="text-davinci-003",
    model_name="text-davinci-003",
    openai_api_base=os.getenv("GPT_BASE"),
    openai_api_key=os.getenv("GPT_KEY"),
    openai_api_version=openai_api_version,
    temperature=0)

chat_llm = AzureChatOpenAI(
    deployment_name=chat_deploy,
    model_name=chat_model,
    openai_api_base=os.getenv("GPT_BASE"),
    openai_api_key=os.getenv("GPT_KEY"),
    openai_api_version=openai_api_version,
    temperature=0)
