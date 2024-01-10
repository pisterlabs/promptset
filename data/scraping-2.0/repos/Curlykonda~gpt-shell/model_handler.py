import logging
import os
import time
from typing import Callable, List, Optional

import openai
from openai.error import RateLimitError, ServiceUnavailableError, Timeout

from prompt_loader import get_system_prompt

openai.api_key = os.getenv("OPENAI_API_KEY")

_MAX_RETRIES = 3
_RETRY_TIMEOUT = 10  # seconds
_INVALID_RESPONSE = "INVALID_RESPONSE"

logger = logging.getLogger(__name__)


def _with_retries(api_call: Callable, invalid_response=_INVALID_RESPONSE):
    for _ in range(_MAX_RETRIES):
        try:
            return api_call()
        except openai.APIError:
            logger.warning("API Error. Sleep and try again.")
            time.sleep(_RETRY_TIMEOUT)
        except RateLimitError:
            logger.error("Rate limiting, Sleep and try again.")
            time.sleep(_RETRY_TIMEOUT)
        except Timeout:
            logger.error("Timeout, Sleep and try again.")
            time.sleep(_RETRY_TIMEOUT)
        except ServiceUnavailableError:
            logger.error("Service Unvailable, Sleep and try again.")
            time.sleep(_RETRY_TIMEOUT * 3)
        except KeyError:
            logger.error("Unexpected response format. Sleep and try again.")
            time.sleep(_RETRY_TIMEOUT)

    logger.error("Reached retry limit and did not obtain proper response")
    return invalid_response


def get_chat_completion(
    model: str,
    prompt_turns: List[dict],
    max_tokens=512,
    temperature=0.0,
    stop: Optional[List[str]] = None,
    logit_bias: Optional[dict] = {},
):
    def api_call():
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt_turns,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            logit_bias=logit_bias,
        )

        if "choices" not in response:
            raise KeyError("Response does not contain choices")

        return response["choices"][0]

    return _with_retries(api_call)


if __name__ == "__main__":
    prompt = "Write back 'This is a test'"

    # chat_model = "gpt-3.5-turbo-1106"
    chat_model = "gpt-4"
    prompt_turns = [{"role": "user", "content": prompt}]

    chat_completion = get_chat_completion(chat_model, prompt_turns)
    print("Chat completion: ", chat_completion["message"]["content"])

    prompt = "How do I list all files in my downloads folder?"
    prompt_turns = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": prompt},
    ]
    chat_completion = get_chat_completion(chat_model, prompt_turns)
    print("Chat completion: ", chat_completion["message"]["content"])
