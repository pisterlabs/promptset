import os
from langchain.llms import AzureOpenAI

def setupAzureOpenAI():
  from dotenv import load_dotenv
  load_dotenv()

  return AzureOpenAI(
     openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
     model_kwargs={
        "api_type": os.getenv("AZURE_OPENAI_API_TYPE"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_base": os.getenv("AZURE_OPENAI_API_BASE"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "deployment_id": "turbo-0301"
     },
     model_name="gpt-3.5-turbo")