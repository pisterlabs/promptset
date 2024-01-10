from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.llms.openai import AzureOpenAI

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient


SEARCHCLIENT = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY")),
    index_name="chatbot-search-meta-v0",
)

CHAT_LLM = AzureChatOpenAI(
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_type=os.getenv("OPENAI_API_TYPE"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    model_name="gpt-35-turbo",
    deployment_name="gpt-35-turbo",
    temperature=0,
    model_version="0301",
    # model_kwargs={"streaming": True},
)

CHAT_LLM_4 = AzureChatOpenAI(
    openai_api_base=os.getenv("GPT4_OPENAI_API_BASE"),
    openai_api_key=os.getenv("GPT4_OPENAI_API_KEY"),
    openai_api_type=os.getenv("GPT4_OPENAI_API_TYPE"),
    openai_api_version=os.getenv("GPT4_OPENAI_API_VERSION"),
    model_name="gpt-4-32k",
    deployment_name="gpt-4-32k",
    temperature=0,
    # model_kwargs={"streaming": True},
)

LLM = AzureOpenAI(
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_type=os.getenv("OPENAI_API_TYPE"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    model_name="text-davinci-003",
    deployment_name="text-davinci-003",
    temperature=0,
)

EMBEDDING = OpenAIEmbeddings(deployment="text-embedding-ada-002")


from langchain.vectorstores.azuresearch import AzureSearch

def vector_setting():
  vector_store: AzureSearch = AzureSearch(
      azure_search_endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
      azure_search_key=os.getenv("AZURE_SEARCH_ADMIN_KEY"),
      index_name="chatbot-search-meta-v0",
      embedding_function=EMBEDDING.embed_query,
  )
  return vector_store