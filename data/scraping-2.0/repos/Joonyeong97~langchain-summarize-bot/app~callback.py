"""Callback handlers used in the app."""

from langchain.callbacks.base import AsyncCallbackHandler



class LLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket