import os
import openai
import dotenv

from langchain.llms import AzureOpenAI
from langchain.chat_models import ChatOpenAI

from typing import Any, Optional

def get_llm(temperature=0.0, top_p=1, max_tokens=2000, deployment=None, model=None):
    # load environment variables using dotenv
    dotenv.load_dotenv()

    print(f"AZURE_OPENAI_ENDPOINT={os.getenv('AZURE_OPENAI_ENDPOINT')}")

    AZURE_OPENAI_ENABLED = os.getenv("AZURE_OPENAI_ENABLED").lower() == "true"

    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

    AZURE_MODE = os.getenv("AZURE_MODE")
    AZURE_OPENAI_DEPLOYMENT_NAME = deployment if deployment is not None \
        else os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    AZURE_OPENAI_MODEL_NAME = model if model is not None \
        else os.getenv("AZURE_OPENAI_MODEL_NAME")

    # print all those env vars, except for the keys
    print(f"AZURE_OPENAI_ENABLED={AZURE_OPENAI_ENABLED}")
    print(f"RESOURCE_ENDPOINT={RESOURCE_ENDPOINT}")
    print(f"AZURE_MODE={AZURE_MODE}")
    print(f"AZURE_OPENAI_DEPLOYMENT_NAME={AZURE_OPENAI_DEPLOYMENT_NAME}")
    print(f"AZURE_OPENAI_MODEL_NAME={AZURE_OPENAI_MODEL_NAME}")

    if AZURE_OPENAI_ENABLED:
        openai.api_type = "azure"
        openai.api_key = AZURE_OPENAI_API_KEY
        openai.api_base = RESOURCE_ENDPOINT
        openai.api_version = "2023-03-15-preview"

    # url = openai.api_base + "/openai/deployments?api-version=2022-12-01"

    if not AZURE_OPENAI_ENABLED:
        llm = ChatOpenAI(model_name=AZURE_OPENAI_MODEL_NAME,
                         temperature=temperature, top_p=top_p)
    else:
        llm = ChatOpenAI(model=AZURE_OPENAI_MODEL_NAME,
                            temperature=temperature,
                            max_tokens=max_tokens if max_tokens != -1
                            else None,
                            openai_api_base=openai.api_base,
                            openai_api_key=openai.api_key,
                            model_kwargs={
                                "engine": AZURE_OPENAI_DEPLOYMENT_NAME,
                                "top_p": top_p})

    return llm
