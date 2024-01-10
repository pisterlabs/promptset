import os
import openai
from dotenv import load_dotenv, find_dotenv

import warnings

warnings.filterwarnings("ignore")


def load_openai_api_key():
    """Load OpenAI API key from .env file."""
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]
