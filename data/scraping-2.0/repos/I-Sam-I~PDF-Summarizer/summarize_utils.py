# OpenAI needs collections.abc
import collections 
try:
    from collections.abc import MutableSet, MutableMapping
    collections.MutableSet = MutableSet
    collections.MutableMapping = MutableMapping
except AttributeError:
    from collections import MutableSet, MutableMapping

from dotenv import load_dotenv
from os import getenv
import openai
from time import sleep

import logging

# Create a custom logger
logger = logging.getLogger(__name__)

# Load the OpenAI API key from .env
load_dotenv()
openai.api_key = getenv("OPENAI_API_KEY")

def summarize_text(text: list[str]) -> list[str]:
    """
    Summarize the entire text.

    Args:
        text (list[str]): The entire text to summarize.

    Returns:
        list[str]: The summarized text.
    """
    max_retries = 6
        
    responses: list[str] = []
    for page in text:
        for i in range(max_retries):
            try: 
                responses.append(get_gpt_response(page))
                break
        
            except openai.RateLimitError:
                wait_time = 2 ** i # Exponential backoff
                logger.warning(f"OpenAI API rate limit reached, retrying after {wait_time} seconds.")
                sleep(wait_time)
    
    return responses


def get_gpt_response(text: str) -> str:
    """
    Connects with the OpenAI API and returns the response.

    Args:
        text (str): The text to summarize.

    Returns:
        str: Summarized text.
    """
     
    return openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Summarize the content you are given for a high school student. The text is {text}",
        temperature=0,
        max_tokens=4097 - len(text) // 3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    ).choices[0].text.strip()
