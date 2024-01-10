from typing import Annotated, Optional

from fastapi import Depends, Header
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.llms.huggingface_text_gen_inference import (
    HuggingFaceTextGenInference,
)
from langchain_core.language_models import BaseLLM
from langchain_core.memory import BaseMemory

from pybot.callbacks import TracingLLMCallbackHandler
from pybot.config import settings
from pybot.history import ContextAwareMessageHistory
from pybot.prompts.chatml import AI_PREFIX, AI_SUFFIX, HUMAN_PREFIX


def UserIdHeader(alias: Optional[str] = None, **kwargs):
    if alias is None:
        alias = settings.user_id_header
    return Header(alias=alias, **kwargs)


def UsernameHeader(alias: Optional[str] = None, **kwargs):
    if alias is None:
        alias = "X-Forwarded-Preferred-Username"
    return Header(alias=alias, **kwargs)


def EmailHeader(alias: Optional[str] = None, **kwargs):
    if alias is None:
        alias = "X-Forwarded-Email"
    return Header(alias=alias, **kwargs)


def MessageHistory() -> RedisChatMessageHistory:
    return ContextAwareMessageHistory(
        url=str(settings.redis_om_url),
        key_prefix="pybot:messages:",
        session_id="sid",  # a fake session id as it is required
    )


def ChatMemory(
    history: Annotated[RedisChatMessageHistory, Depends(MessageHistory)]
) -> BaseMemory:
    return ConversationBufferWindowMemory(
        human_prefix=HUMAN_PREFIX,
        ai_prefix=AI_PREFIX,
        memory_key="history",
        input_key="input",
        output_key="output",
        chat_memory=history,
        return_messages=True,
    )


def Llm() -> BaseLLM:
    return HuggingFaceTextGenInference(
        inference_server_url=str(settings.inference_server_url),
        max_new_tokens=1024,
        temperature=None,
        # top_p=0.9,
        stop_sequences=[
            AI_SUFFIX
        ],  # not all mistral models have a decent tokenizer config.
        streaming=True,
        callbacks=[TracingLLMCallbackHandler()],
    )
