import time
import openai
from halo import Halo
from openai import OpenAI
from .ui_utils import logger


def check_api_key(api_key):
    """
    Validates the provided OpenAI API key.

    This function attempts to list the models using the given API key to check its validity.
    It uses a spinner to indicate progress and logs any authentication errors encountered.

    Args:
        api_key (str): The API key to be validated.

    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    spinner = Halo(text="Checking API key", spinner="dots")
    spinner.start()

    client = OpenAI(api_key=api_key)

    try:
        # Attempt to list models to verify the API key.
        client.models.list()
    except openai.AuthenticationError as e:
        spinner.fail("Invalid API key!")
        logger.error(e)  # Log the error for debugging purposes.
        return False
    else:
        spinner.succeed("API key is valid 🎉")
        time.sleep(1)  # Short pause for user readability.
        return True
