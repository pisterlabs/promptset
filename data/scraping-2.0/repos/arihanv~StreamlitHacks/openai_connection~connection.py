import streamlit as st
from streamlit.connections import ExperimentalBaseConnection
from streamlit.runtime.caching import cache_data
import openai


class OpenAIConnection(ExperimentalBaseConnection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_tokens = 0

    def get_token_count(self):
        return self.total_tokens

    def _connect(self, **kwargs):
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = st.secrets["openai_api_key"]
        openai.api_key = api_key
        return openai

    def query(
        self,
        query: str,
        model: str = "gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant"}],
        ttl: int = 3600,
        **kwargs
    ) -> dict:
        @cache_data(ttl=ttl)
        def _query(query: str, model: str, messages: list, **kwargs) -> dict:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages + [{"role": "user", "content": query}],
            )
            self.total_tokens += response["usage"]["total_tokens"]
            return response

        return _query(query, model, messages, **kwargs)
