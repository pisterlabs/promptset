## IMPORT NECESSARY PACKAGES 
import os
import environ
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms import AzureOpenAI
# import openai

## LOAD .ENV FILE FOR API CONNECTION
env = environ.Env()

# init Azure OpenAI API
environ.Env.read_env()
OPENAI_API_KEY = env("OPENAI_API_KEY")
AZURE_API_KEY = env("AZURE_API_KEY")
OPENAI_DEPLOYMENT_ENDPOINT = env("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_VERSION = env("OPENAI_DEPLOYMENT_VERSION")
OPENAI_DEPLOYMENT_NAME = env("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = env("OPENAI_MODEL_NAME")
# OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = env("OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME")
# OPENAI_ADA_EMBEDDING_MODEL_NAME = env("OPENAI_ADA_EMBEDDING_MODEL_NAME")



# openai.api_type = "azure"
# openai.api_version = OPENAI_DEPLOYMENT_VERSION
# openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
# openai.api_key = AZURE_API_KEY


# model configuration
temp = 0.1
llm = AzureChatOpenAI(
    openai_api_key = AZURE_API_KEY,
    openai_api_base = OPENAI_DEPLOYMENT_ENDPOINT,
    openai_api_version = OPENAI_DEPLOYMENT_VERSION,
    deployment_name = OPENAI_DEPLOYMENT_NAME, 
    model_name = OPENAI_MODEL_NAME,
    temperature = temp,
    )



