import os.path as osp

import openai

from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI


def load_api_key(api_key_filename: str) -> str:
    """
    Attempts to load a locally saved API key, if it
    does not exist we prompt the user to give one.

    Args:
        api_key_filename (str): indicates the file where the API key would be saved locally

    Returns:
        str: OpenAI API key
    """
    if not osp.exists(api_key_filename):
        with open(api_key_filename, "w", encoding="utf-8") as file:
            file.write("")

    with open(api_key_filename, "r", encoding="utf-8") as file:
        api_key: str = file.read()
        # If the API key file is empty, we prompt the user to give theirs
        if not api_key:
            print(
                "To start the app, please provide your OpenAI API key"
                + " (https://platform.openai.com/api-keys):"
            )
            api_key = input()
        return api_key


def validate_api_key(api_key: str) -> bool:
    """Validates a key by using it to prompt an LLM.

    Args:
        api_key (str): OpenAI API key

    Returns:
        bool: returns true if the key is valid
    """
    try:
        chat_model: ChatOpenAI = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=api_key)
        chat_model.predict(
            "Only reply yes or no"
        )
        return True

    except openai.AuthenticationError:
        error_message: str = "ERROR:  API Key is invalid"
        print(f"\033[91m{error_message}\033[0m")
        return False


def retrieve_api_key() -> str:
    """
    Retrieves an API key from a local file or by prompting the user,
    keeps trying until a valid key is provided.

    Returns:
        str: OpenAI API key
    """
    api_key_filename: str = "openai_api_key.txt"

    api_key: str = load_api_key(api_key_filename)

    if validate_api_key(api_key):
        with open(api_key_filename, "w", encoding="utf-8") as file:
            file.write(api_key)
    else:
        # If no valid API key was provided, we try again
        return retrieve_api_key()

    return api_key
