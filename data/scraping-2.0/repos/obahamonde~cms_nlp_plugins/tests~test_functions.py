import pytest
from openai import AsyncOpenAI

from backend import use_chat


# Mocking the AsyncOpenAI object
@pytest.fixture
def ai():
    return AsyncOpenAI()


@pytest.mark.asyncio
async def test_use_chat(ai):
    test_result = await use_chat(ai=ai, text="Hello, world!")
    assert isinstance(test_result, str)
