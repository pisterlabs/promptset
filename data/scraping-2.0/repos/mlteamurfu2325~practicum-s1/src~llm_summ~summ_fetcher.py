"""Module for fetching summary from LLM API for provided text."""
import os

import openai


def fetch_summary(
    text: str,
    llm_model: str = "openai/gpt-4-1106-preview",
    llm_api_key: str | None = None,
) -> str:
    """Fetch summary from LLM API using provided text and optionally an API key.

    :param text: Input text to summarize
    :type text: str
    :param llm_model: Name of LLM model to use
    :type llm_model: str
    :param llm_api_key: API key for service (optional)
    :type llm_api_key: str, optional
    :return: Summary text response from API
    :rtype: str

    :Example:

    >>> summary = fetch_summary(text)
    >>> summary = fetch_summary(api_key, text)

    """
    try:
        if os.environ.get("LLM_API_KEY", None):
            openai.api_key = os.environ["LLM_API_KEY"]
        else:
            openai.api_key = llm_api_key
    except Exception as error:
        raise ValueError(f"LLM_API key error: {error}") from error

    try:
        openai.base_url = os.environ["LLM_URL"]
    except Exception as error:
        raise ValueError(f"LLM_URL error: {error}") from error

    prompt = f"Дай краткий пересказ этого текста: {text}"

    messages = []

    messages.append({"role": "user", "content": prompt})

    response_big = openai.chat.completions.create(
        model=llm_model,
        messages=messages,
        temperature=0.7,
        n=1,
        max_tokens=int(len(prompt) * 1.5),
    )

    response = response_big.choices[0].message.content

    return response
