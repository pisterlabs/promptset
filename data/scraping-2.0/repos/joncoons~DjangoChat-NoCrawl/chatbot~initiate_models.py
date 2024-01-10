import requests
from requests import get, post
import os
import openai
from dotenv import load_dotenv

load_dotenv()

def setModels():
    AOAI_API_KEY = os.getenv("AOAI_API_KEY") 
    AOAI_URI = os.getenv("AOAI_URI") 
    AOAI_API_VERSION = os.getenv("AOAI_API_VERSION")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    COMPLETION_MODEL = os.getenv("COMPLETION_MODEL")
    CHATGPT_MODEL = os.getenv("CHATGPT_MODEL")
    openai.api_type = "azure"
    openai.api_key = AOAI_API_KEY
    openai.api_base = AOAI_URI
    openai.api_version = AOAI_API_VERSION
    aoai_url = openai.api_base + "/openai/deployments?api-version=2023-05-15"

    headers= {
        "api-key": AOAI_API_KEY,
        "Accept": 'text/event-stream'
    }

    r = requests.get(aoai_url, headers=headers)
    if r:
        print("Azure Open AI Models Instantiated")

def setFormRecognizer():
    FR_ENDPOINT = os.getenv("FR_ENDPOINT")
    FR_KEY = os.getenv("FR_KEY")  
    post_url = FR_ENDPOINT + "/formrecognizer/v2.1/layout/analyze"
    headers = {
        'Content-Type': 'application/pdf',
        'Ocp-Apim-Subscription-Key': FR_KEY,
    }