import os

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASSISTANT_ID = os.getenv('ASSISTANT_ID')

ERROR_MESSAGE = 'We are facing an issue at this moment.'

client = OpenAI(
    api_key=OPENAI_API_KEY
)
