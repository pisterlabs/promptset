import os

from dotenv import load_dotenv
import pytest
import openai


def test_open_ai_response():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    try:
        openai.Completion.create(
          engine="davinci",
          prompt="test",
          temperature=0.9,
          max_tokens=150,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0.6,
          stop=["\n", " Human:", " AI:"]
        )

    except:
        pytest.fail("An error occurred with OpenAI request")

