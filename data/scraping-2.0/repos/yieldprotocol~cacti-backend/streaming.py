from typing import Any, Callable

from text_generation import Client
from langchain.llms import OpenAI, HuggingFaceTextGenInference
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import initialize_agent
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import config
from agents.conversational import CUSTOM_AGENT_NAME
from chains import IndexAPIChain
from utils.constants import HUGGINGFACE_API_KEY, HUGGINGFACE_INFERENCE_ENDPOINT


class StreamingCallbackHandler(StreamingStdOutCallbackHandler):
    """Override the minimal handler to get the token."""

    def __init__(self, new_token_handler: Callable) -> None:
        self.new_token_handler = new_token_handler

    @property
    def always_verbose(self) -> bool:
        return True

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.new_token_handler(token)


def get_streaming_llm(new_token_handler, model_name=None, max_tokens=None):
    if model_name=='huggingface-llm':
        # falls back to non-streaming if none provided
        streaming_kwargs = dict(
            stream=True,
            callbacks=[StreamingCallbackHandler(new_token_handler)],
        ) if new_token_handler else {}

        inference_server_url = HUGGINGFACE_INFERENCE_ENDPOINT
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json",
        }
        client = Client(inference_server_url, headers=headers)
        llm = HuggingFaceTextGenInference(
            inference_server_url=inference_server_url,
            max_new_tokens=max_tokens if max_tokens is None else 200,
            temperature=0.1, # should be strictly positive
            **streaming_kwargs,
        )
        llm.client = client
    else:
        # falls back to non-streaming if none provided
        streaming_kwargs = dict(
            streaming=True,
            callbacks=[StreamingCallbackHandler(new_token_handler)],
        ) if new_token_handler else {}

        model_kwargs = dict(
            model_name=model_name,
        ) if model_name else {}

        if not model_name or model_name in ('text-davinci-003',):
            model_cls = OpenAI
            if max_tokens is None:
                max_tokens = -1  # this is setting for unlimited
        else:
            model_cls = ChatOpenAI

        llm = model_cls(
            temperature=0.0,
            max_tokens=max_tokens,
            **streaming_kwargs,
            **model_kwargs,
        )
    return llm


def get_streaming_chain(prompt, new_token_handler, use_api_chain=False, model_name=None, max_tokens=None):
    llm = get_streaming_llm(new_token_handler, model_name=model_name, max_tokens=max_tokens)

    if use_api_chain:
        return IndexAPIChain.from_llm(
            llm,
            verbose=True
        )
    else:
        return LLMChain(llm=llm, prompt=prompt, verbose=True)


def get_streaming_tools(tools, new_token_handler):
    streaming_tools = config.initialize_streaming(tools, new_token_handler)
    return streaming_tools


def get_streaming_agent(tools, new_token_handler, model_name=None, **agent_kwargs):
    llm = get_streaming_llm(new_token_handler, model_name=model_name)
    agent = initialize_agent(tools, llm, agent=CUSTOM_AGENT_NAME, **agent_kwargs)
    return agent
