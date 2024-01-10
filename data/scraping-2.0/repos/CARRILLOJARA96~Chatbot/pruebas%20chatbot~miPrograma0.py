import os
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key
try:
    modelos = openai.Model.list()
    print(modelos)
except Exception as e:
    print(f"Error: {e}")
##modelos = openai.Model.list()
##print(modelos)

