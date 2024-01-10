import os
import pytest
import openai
from unittest.mock import patch

from infraestructure.adapter.openai_adapter import OpenAIAdapter
from infraestructure.adapter.openai_start import login_openai

openai_adapter = OpenAIAdapter()


def test_login_openai():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "my_key",
            "OPENAI_API_BASE": "my_base",
            "OPENAI_API_TYPE": "my_type",
            "OPENAI_API_VERSION": "my_version",
        },
    ):
        login_openai()
        assert openai.api_key == "my_key"
        assert openai.api_base == "my_base"
        assert openai.api_type == "my_type"
        assert openai.api_version == "my_version"


def test_ask_openai():
    login_openai()
    question = "¿Cuál es la capital de Francia?"
    text = ""
    task = "question"
    response = openai_adapter.ask_openai(question, text, task)
    assert len(response) > 0
    assert isinstance(response, str)
