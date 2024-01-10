import os
import openai
from dotenv import load_dotenv
from elevenlabs import set_api_key

load_dotenv()
load_dotenv("/var/www/py/plato/server")

# Fetch API keys from .env file
OPENAI_KEY = os.getenv('OPENAI_KEY')
ELEVENLABS_KEY = os.getenv('ELEVENLABS_KEY')
ACCESS_CODE = os.getenv('ACCESS_CODE')

# Set API keys
set_api_key(ELEVENLABS_KEY)
openai.api_key = OPENAI_KEY
