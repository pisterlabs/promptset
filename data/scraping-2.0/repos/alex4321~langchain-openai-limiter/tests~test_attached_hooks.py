import os
from .utils import load_env
import pytest
import time
from langchain_openai_limiter.limit_info import get_limit_info, reset_limit_info
from langchain_openai_limiter.capture_headers import attach_session_hooks
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


def test_attach_sync_session_hooks(load_env):
    reset_limit_info()
    attach_session_hooks()
    api_key = os.environ["OPENAI_API_KEY"]
    chat_model = ChatOpenAI(
        model_name="gpt-4-0613",
        openai_api_key=api_key,
    )
    assert get_limit_info("gpt-4-0613", api_key) is None
    history = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        ),
    ]
    chat_model(history)
    limit_info = get_limit_info("gpt-4-0613", api_key)
    assert limit_info is not None


@pytest.mark.asyncio
async def test_attach_async_session_hooks(load_env):
    reset_limit_info()
    attach_session_hooks()
    api_key = os.environ["OPENAI_API_KEY"]
    chat_model = ChatOpenAI(
        model_name="gpt-4-0613",
        openai_api_key=api_key,
    )
    assert get_limit_info("gpt-4-0613", api_key) is None
    history = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        ),
    ]
    await chat_model.ainvoke(history)
    limit_info = get_limit_info("gpt-4-0613", api_key)
    assert limit_info is not None
