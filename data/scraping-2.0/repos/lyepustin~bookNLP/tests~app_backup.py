# File: LangChainchatOpenAI.py
# Author: Denys L
# Date: October 8, 2023
# Description:


import os
import sys
from typing import Any
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from fundamentals.langchain_utils import StuffSummarizerByChapter


class StreamingStdOutCallbackHandlerPersonal(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        st.session_state.full_response = st.session_state.full_response + token
        st.session_state.placeholder.markdown(
            st.session_state.full_response + "▌")
        sys.stdout.write(token)
        sys.stdout.flush()


def handle_question(prompt):
    st.session_state.full_response = ""
    st.session_state.handler_ia_message = st.chat_message(
        "assistant", avatar="🤖")
    st.session_state.placeholder = st.session_state.handler_ia_message.empty()
    st.session_state.llm.summarize(os.getenv("BOOK_PATH"))
    st.session_state.placeholder.markdown(st.session_state.full_response)


def main():
    load_dotenv()
    st.title("ChatGPT-like storyteller")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.handler = StreamingStdOutCallbackHandlerPersonal()
        st.session_state.llm = StuffSummarizerByChapter(
            st.session_state.handler)

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "avatar": "🧑‍💻"})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        handle_question(prompt)

        st.session_state.messages.append(
            {"role": "assistant", "content": st.session_state.full_response, "avatar": "🤖"})
        st.session_state.full_response = ""


if __name__ == '__main__':
    main()
