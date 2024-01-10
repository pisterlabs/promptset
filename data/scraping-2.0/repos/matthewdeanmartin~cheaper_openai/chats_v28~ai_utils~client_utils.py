import os

import openai
from dotenv import load_dotenv

load_dotenv()


def create_client():
    openai.api_key = os.environ["OPENAI_API_KEY"]
