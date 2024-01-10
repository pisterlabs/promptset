"""Summarization using GTP-3."""

import os
from typing import Any, Literal, Optional

import openai
from pydantic import BaseModel, PositiveFloat

from src.text_utils import word_count

OpenaiGpt3Engine = Literal["davinci", "curie", "babbage", "ada"]


class Gpt3SummarizationConfiguration(BaseModel):
    """GPT-3 configuration parameters."""

    engine: OpenaiGpt3Engine = "davinci"
    temperature: PositiveFloat = 0.3
    max_ratio: PositiveFloat = 0.3
    top_p: Optional[PositiveFloat] = None
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.1


def _openai_api_key() -> None:
    if (openai_key := os.getenv("OPENAI_API_KEY")) is None:
        raise BaseException("OpenAI API key not found.")
    openai.api_key = openai_key
    return None


def _text_to_gpt3_prompt(text: str) -> str:
    prefix = 'Summarize the following scientific article:\n"""\n'
    suffix = '\n"""\nSummary:\n"""\n'
    prompt = prefix + text + suffix
    return prompt


def _extract_gpt3_result(gpt3_response: dict) -> str:
    return gpt3_response["choices"][0]["text"]


def summarize(text: str, config_kwargs: dict[str, Any]) -> str:
    """Summarize text using GPT-3.

    Args:
        text (str): Input text.
        config_kwargs (dict[str, Any]): GPT-3 configuration parameters.

    Returns:
        str: Summarized test.
    """
    _openai_api_key()
    config = Gpt3SummarizationConfiguration(**config_kwargs)
    prompt = _text_to_gpt3_prompt(text)
    res = openai.Completion.create(
        prompt=prompt,
        engine=config.engine,
        temperature=config.temperature,
        max_tokens=int(word_count(text) * config.max_ratio),
        top_p=config.top_p,
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty,
        stop=['"""'],
    )
    text_sum = _extract_gpt3_result(res)
    return text_sum
