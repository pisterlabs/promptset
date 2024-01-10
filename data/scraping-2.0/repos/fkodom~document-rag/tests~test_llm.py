from typing import Optional, Type

import pytest

from document_rag.llm import BaseLLM, load_llm
from document_rag.llm.huggingface import HuggingFaceLLM


@pytest.mark.parametrize(
    "type, model, error",
    [
        ("openai", "gpt-3.5-turbo-1106", None),
        ("openai", "model-does-not-exist", ValueError),
        ("huggingface", "distilgpt2", None),
        ("huggingface", "model-does-not-exist", OSError),  # error from HF backend
        ("unsupported-type", "distilgpt2", ValueError),
    ],
)
def test_load_llm(type: str, model: str, error: Optional[Type[Exception]]):
    if error is not None:
        with pytest.raises(error):
            _ = load_llm(type=type, model=model)
    else:
        _ = load_llm(type=type, model=model)


@pytest.fixture(scope="session")
def llm() -> HuggingFaceLLM:
    """NOTE: I'm avoiding testing OpenAILLM, because it incurs charges from OpenAI.
    These tests run on every 'git push', and I don't want to incur charges every time.
    For now, if Mypy doesn't throw any type errors for the OpenAI code, just assume
    that the OpenAI API works as expected.
    """
    return HuggingFaceLLM(model="distilgpt2")


def test_generate(llm: BaseLLM):
    _ = llm.generate(prompt="Respond with just the word STOP.")
