import openai
import os
from dotenv import load_dotenv

load_dotenv()

def initialize_openai():
  openai.api_type = 'azure'
  openai.api_base = os.environ.get("OPENAI_API_BASE")
  openai.api_version = os.environ.get("OPENAI_API_VERSION")
  openai.api_key = os.environ.get("OPENAI_API_KEY")