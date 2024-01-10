import requests

import streamlit as st
from langchain.schema import HumanMessage, AIMessage

from src import CFG
from src.codellama import get_prompt
from streamlit_app import get_http_status

API_URL = f"http://{CFG.HOST}:{CFG.PORT_CODELLAMA}"


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="code_assistant")
    if clear_button or "ca_messages" not in st.session_state:
        st.session_state.ca_messages = []


def get_answer(inputs: str) -> str:
    payload = {"inputs": get_prompt(inputs)}
    headers = {"Content-Type": "application/json"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()["content"]


def code_assistant():
    st.sidebar.title("Code Assistant")
    st.sidebar.info("Code Assistant is powered by CodeLlama.")
    st.sidebar.info(f"Running on {CFG.DEVICE}")
    get_http_status(API_URL)

    init_messages()

    # Display chat history
    for message in st.session_state.ca_messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    if user_input := st.chat_input("Your input"):
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.ca_messages.append(HumanMessage(content=user_input))

        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                answer = get_answer(user_input)
            st.markdown(answer)

        st.session_state.ca_messages.append(AIMessage(content=answer))
