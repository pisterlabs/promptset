import openai
import time
import json
import config.settings
import random

from utils.logger import logger  

openai.api_key = config.settings.OPENAI_API_KEY

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = config.settings.OPENAI_MAX_RETRY_COUNT,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    logger.error("Max retry count reached. Raising error.")
                    raise
                delay *= exponential_base * (1 + jitter * random.random())
                logger.info(f"Sleeping for {delay} seconds after {e}...")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
    return wrapper

@retry_with_exponential_backoff
def call_openai_with_backoff(messages):
    logger.debug(f"Calling OpenAI API with model: {config.settings.LANGUAGE_MODEL} and messages: {messages}")
    response = openai.ChatCompletion.create(
        model=config.settings.LANGUAGE_MODEL,
        messages=messages
    )
    return json.dumps(response["choices"][0]["message"])

# Function to call OpenAI API
def call_openai(messages):
    """
    Calls the OpenAI API to get an intelligent response for query processing.
    """
    return call_openai_with_backoff(messages)
