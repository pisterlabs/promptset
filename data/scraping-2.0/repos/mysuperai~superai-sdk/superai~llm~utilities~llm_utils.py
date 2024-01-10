import time
from functools import wraps

import openai
from openai.error import RateLimitError
from requests.exceptions import ConnectionError

from superai.llm.foundation_models.openai import ChatGPT


def call_ai_function(function: str, args: list, description: str, foundation_model=None) -> str:
    """Call an AI function
    This is a magic function that can do anything with no-code. See
    https://github.com/Torantulino/AI-Functions for more info.
    Args:
        function (str): The function to call
        args (list): The arguments to pass to the function
        description (str): The description of the function
        model (str, optional): The model to use. Defaults to None.
    Returns:
        str: The response from the function
    """

    if foundation_model is None:
        foundation_model = ChatGPT()

    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    args = ", ".join(args)
    messages = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}"
            f"\n{function}```\n\nOnly respond with your `return` value.",
        },
        {"role": "user", "content": args},
    ]

    return foundation_model.predict(messages)


def retry(
    func,
    max_retries=3,
    retry_factor=2,
    retry_min_timeout=1000,
    retry_max_timeout=10000,
    allowed_exceptions=(ConnectionError, RateLimitError),
):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        retries = max_retries
        while retries > 0:
            try:
                return func(self, *args, **kwargs)
            # except allowed_exceptions as e:
            except Exception as e:
                print(e)
                if retries == 1:
                    raise e
                timeout = min(retry_max_timeout, retry_min_timeout * (retry_factor ** (max_retries - retries)))
                time.sleep(timeout / 1000)
                retries -= 1

    return wrapper


def check_open_ai_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
    except Exception as e:
        raise Exception("Invalid API key. Error: " + str(e))
