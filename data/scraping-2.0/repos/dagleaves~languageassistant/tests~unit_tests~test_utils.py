import os

import openai
import pytest

from languageassistant.utils import load_openai_api_key

invalid_openai_api_key = os.getenv("OPENAI_API_KEY") in [None, "", "api_key"]


@pytest.mark.skipif(invalid_openai_api_key, reason="Needs valid OpenAI API key")
def test_openai_api_key_env_success() -> None:
    load_openai_api_key()
    assert openai.api_key != ""


def test_openai_api_key_string_success() -> None:
    load_openai_api_key("api_key")
    assert openai.api_key != ""
