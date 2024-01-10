
# Import from standard library
import logging

# Import from 3rd party libraries
import openai
from dotenv import load_dotenv
import os
load_dotenv()

# Assign credentials from environment variable or streamlit secrets dict
openai.api_key = os.environ.get("API_KEY_1")

# Suppress openai request/response logging
# Handle by manually changing the respective APIRequestor methods in the openai package
# Alternatively (affects all messages from this logger):
logging.getLogger("openai").setLevel(logging.WARNING)


class Openai:
    """OpenAI Connector."""

    @staticmethod
    def moderate(prompt: str) -> bool:
        """Call OpenAI GPT Moderation with text prompt.
        Args:
            prompt: text prompt
        Return: boolean if flagged
        """
        try:
            response = openai.Moderation.create(prompt)
            return response["results"][0]["flagged"]

        except Exception as e:
            logging.error(f"OpenAI API error: {e}")

    @staticmethod
    def complete(prompt: str, temperature: float = 0.9, max_tokens: int = 1000) -> str:
        """Call OpenAI GPT Completion with text prompt.
        Args:
            prompt: text prompt
        Return: predicted response text
        """
        kwargs = {
            "engine": "text-davinci-003",
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,  # default
            "frequency_penalty": 0,  # default,
            "presence_penalty": 0,  # default
        }
        try:
            response = openai.Completion.create(**kwargs)
            return response["choices"][0]["text"]

        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
