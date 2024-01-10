from typing import Any, Optional
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import logging
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    # def on_llm_end(self, *args, **kwargs):
        # st.session_state["msg"].append({"msg": self.message, "role": "ai"})

    # def on_llm_new_token(self, token, *args, **kwargs):
        # self.message += token
        # self.message_box.markdown(self.message)
    
    def on_llm_error(self, token, *args, **kwargs):
        logging.error(f'on_llm_error -> ${token}')