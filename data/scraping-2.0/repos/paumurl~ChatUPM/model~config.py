import os
import openai

#Load .env
from dotenv import load_dotenv
load_dotenv()

openai_api_base = os.getenv("OPENAI_API_BASE")
openai_api_key =os.getenv("OPENAI_API_KEY")
embedding_model = os.getenv('DEPLOYMENT_NAME_EMB')
gpt_model = os.getenv("DEPLOYMENT_NAME_GPT")
deployment_name = os.getenv('DEPLOYMENT_NAME_CHAT')

openai.api_version = "2023-03-15-preview"
openai.api_type = "azure"
openai.api_base = openai_api_base
openai.api_key = openai_api_key
