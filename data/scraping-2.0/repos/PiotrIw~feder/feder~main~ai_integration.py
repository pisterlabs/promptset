import logging

import openai
import tiktoken
from django.conf import settings
from django.utils.translation import gettext_lazy as _

logger = logging.getLogger(__name__)


def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_openai_completion(
    prompt, role="user", temperature=settings.OPENAI_LLM_TEMPERATURE
):
    """
    Uses OpenAI's ChatCompletion API to generate a response to a given prompt.

    Args:
        prompt (str): The prompt to generate a response for.
        role (str, optional): The role of the speaker. Defaults to "user".
        temperature (float, optional): The degree of randomness of the model's output.
            Defaults to settings.OPENAI_LLM_TEMPERATURE.

    Returns:
        str: The generated response.
    """
    model = settings.OPENAI_LLM_MODEL
    if (
        num_tokens_from_string(prompt, model=settings.OPENAI_LLM_MODEL)
        > settings.OPENAI_LLM_MODEL_MAX_TOKENS
    ):
        model = settings.OPENAI_LLM_MODEL_LARGE
    if (
        num_tokens_from_string(prompt, model=settings.OPENAI_LLM_MODEL_LARGE)
        > settings.OPENAI_LLM_MODEL_LARGE_MAX_TOKENS
    ):
        token_count = num_tokens_from_string(
            prompt, model=settings.OPENAI_LLM_MODEL_LARGE
        )
        return _(f"Prompt of {token_count} is too long.")
    openai.api_key = settings.OPENAI_API_KEY
    messages = [{"role": role, "content": prompt}]
    success = False
    retry_count = 0
    response_text = ""
    while not success and retry_count < settings.OPENAI_MAX_RETRIES:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            success = True
            logger.info(f"OpenAI completion response: {response}")
            response_text = response.choices[0].message["content"]
        except Exception as e:
            logger.error(e)
            retry_count += 1
            logger.info(f"OpenAI completion retry no: {retry_count}")
    return response_text
