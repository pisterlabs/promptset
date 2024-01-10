import logging
import os
from typing import Dict, List, Tuple

import openai

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS_DIR = os.path.join(ROOT_DIR, ".keys")
DEFAULT_LLM: str = "gpt-3.5-turbo"
# DEFAULT_LLM: str = "gpt-4"

log = logging.getLogger(__name__)


def set_openai_key(key=None) -> str:
    """
    This function sets the OpenAI API key.

    Parameters:
    key (str): The OpenAI API key. If not provided, the function will try to read it from a file.

    Returns:
    str: The OpenAI API key.

    Raises:
    FileNotFoundError: If the key file is not found and no key is provided.
    """
    if key is None:
        try:
            # Try to read the key from a file
            with open(os.path.join(KEYS_DIR, "openai.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("OpenAI API key not found.")
            return
    # Set the OpenAI API key
    openai.api_key = key
    log.info("OpenAI API key set.")
    return key


def gpt_text(
    messages: List[Dict[str, str]] = None,
    model=DEFAULT_LLM,
    temperature: float = 0,
    max_tokens: int = 32,
    stop: List[str] = ["\n"],
) -> str:
    log.debug(f"Sending messages to OpenAI: {messages}")
    response: Dict = openai.ChatCompletion.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    reply: str = response["choices"][0]["message"]["content"]
    log.debug(f"Received reply from OpenAI: {reply}")
    return reply

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    set_openai_key()
    print(gpt_text(model="gpt-3.5-turbo", max_tokens=8, messages=[{"role": "user", "content": "hello"}]))
