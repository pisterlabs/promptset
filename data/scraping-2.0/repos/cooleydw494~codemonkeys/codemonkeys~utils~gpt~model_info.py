import json
import os
from typing import Optional

import openai

from codemonkeys.defs import TEMP_PATH
from codemonkeys.utils.imports.env import Env
from codemonkeys.utils.misc.handle_exception import handle_exception
from codemonkeys.utils.monk.theme_functions import print_t


def get_gpt_model_info() -> Optional[list[str]]:
    """
    Retrieves cached info on GPT models.

    Attempts to read a JSON file containing cached GPT model information.
    If the file does not exist or contains invalid data, this function will
    gracefully handle the error and return None, allowing the automation flow
    to continue.

    :return: Dictionary containing GPT model information if successful, None otherwise.
    :rtype: Optional[dict]

    :raises FileNotFoundError: If the model info cache file does not exist.
    :raises JSONDecodeError: If the cache file contains invalid JSON.
    """
    try:
        with open(os.path.join(TEMP_PATH, 'model_info_cache.json'), 'r') as f:
            return json.load(f)

    except Exception as e:
        print_t("An error occurred reading cached gpt model info.", 'warning')
        handle_exception(e, always_continue=True)
        return None


def get_gpt_model_names() -> Optional[list[str]]:
    """
    Retrieves the names of all GPT models from the cached model info.

    Extracts and returns a list of model names from the cached model information obtained
    by the `get_gpt_model_info` function. If the model info is not available, it returns None.

    :return: List of model names if model info is available, None otherwise.
    :rtype: Optional[list[str]]
    """
    model_info = get_gpt_model_info()
    if model_info is None:
        return None
    return model_info


def update_gpt_model_cache() -> None:
    """
    Updates the GPT model info cache.

    Contacts the OpenAI API to fetch the latest list of models and updates the
    model info cache file. If there's an error during the update (e.g., an API issue),
    the function will catch the Exception and use the handle_exception utility to manage it.

    :return: None
    """

    try:
        model_info = _query_model_info()
        if model_info is not None:
            with open(os.path.join(TEMP_PATH, 'model_info_cache.json'), 'w') as f:
                model_names = list(model_info.keys())
                json.dump(model_names, f)

    except Exception as e:
        print_t("An error occurred updating GPT model info cache.", 'error')
        handle_exception(e)


def _query_model_info() -> Optional[dict]:
    """
    Queries the OpenAI API for the list of models.

    Private method that makes an API call to OpenAI to fetch a list of current models.
    The response is then shaped into a dictionary mapping model IDs to their respective details.
    It only queries the OpenAI API if the environment variable for the API key is set.

    :return: A dictionary mapping model ids to their details if successful, None otherwise.
    :rtype: Optional[dict]

    :raises Exception: If the OPENAI_API_KEY is not set in the environment variables.
    """

    env = Env.get()
    if env.OPENAI_API_KEY is None:
        raise Exception("OPENAI_API_KEY not set in environment variables")

    client = openai.OpenAI(api_key=env.OPENAI_API_KEY)

    model_list = client.models.list()

    # We're creating a new dictionary where the key is the model's ID and the value is the model's data
    model_info = {model.id: model for model in model_list.data}

    return model_info
