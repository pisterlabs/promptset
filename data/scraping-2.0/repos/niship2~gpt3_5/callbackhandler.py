from langchain.callbacks.base import BaseCallbackHandler
from typing import Any
import streamlit as st


class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """Copied only streaming part from StreamlitCallbackHandler"""

    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)
