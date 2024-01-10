import openai
import os
import colors
from ai import requests
from dotenv import load_dotenv

def auth():
   load_dotenv()
   
   openai.organization = os.getenv("ORG_ID")
   openai.api_key = os.getenv("OPENAI_API_KEY")
   
   openai.Model.list()
   colors.print_success(f"Authenticated with OpenAI API. Using {requests.model}.", False)
   