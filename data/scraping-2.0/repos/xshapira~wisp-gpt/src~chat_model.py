import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory

from src.chat_session_manager import save_message


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


class ChatModel:
    def __init__(
        self,
        llm=None,
        prompt=None,
        memory_llm=None,
        **kwargs,
    ):
        self.llm = llm
        self.prompt = prompt

    def configure_chat_memory(self, memory_llm, **kwargs):
        self.memory_llm = memory_llm

        if "memory" not in st.session_state:
            st.session_state["memory"] = ConversationSummaryBufferMemory(
                llm=self.memory_llm,
                max_token_limit=120,
                memory_key="chat_history",
                return_messages=True,
            )
