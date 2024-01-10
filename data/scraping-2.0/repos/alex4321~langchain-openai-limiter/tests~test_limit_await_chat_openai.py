from langchain_openai_limiter.limit_await_chat_openai import LimitAwaitChatOpenAI
from langchain_openai_limiter.limit_info import reset_limit_info
from langchain_openai_limiter.capture_headers import attach_session_hooks
from .utils import load_env
import os
import pytest
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


def test_limitawait_chat_openai_sync(load_env):
    reset_limit_info()
    api_key = os.environ["OPENAI_API_KEY"]
    chat_model = LimitAwaitChatOpenAI(
        chat_openai=ChatOpenAI(
            model_name="gpt-4-0613",
            openai_api_key=api_key,
        )
    )
    history = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        ),
    ]
    chat_model.generate([history])


@pytest.mark.asyncio
async def test_limitawait_chat_openai_async(load_env):
    reset_limit_info()
    attach_session_hooks()
    api_key = os.environ["OPENAI_API_KEY"]
    chat_model = LimitAwaitChatOpenAI(
        chat_openai=ChatOpenAI(
            model_name="gpt-4-0613",
            openai_api_key=api_key,
        )
    )
    history = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        ),
    ]
    await chat_model.agenerate([history])


def test_limitawait_chat_openai_stream_sync(load_env):
    reset_limit_info()
    attach_session_hooks()
    api_key = os.environ["OPENAI_API_KEY"]
    chat_model = LimitAwaitChatOpenAI(
        chat_openai=ChatOpenAI(
            model_name="gpt-4-0613",
            openai_api_key=api_key,
        )
    )
    history = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        ),
    ]
    for item in chat_model.stream(history):
        pass


@pytest.mark.asyncio
async def test_limitawait_chat_openai_stream_async(load_env):
    reset_limit_info()
    attach_session_hooks()
    api_key = os.environ["OPENAI_API_KEY"]
    chat_model = LimitAwaitChatOpenAI(
        chat_openai=ChatOpenAI(
            model_name="gpt-4-0613",
            openai_api_key=api_key,
        )
    )
    history = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        ),
    ]
    async for item in chat_model.astream(history):
        pass