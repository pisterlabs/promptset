"""Module for building the Langchain Agent"""
import streamlit as st
from langchain.chat_models import ChatOpenAI
from gnosis.agent import PDFExplainer


def build(key, client):
    """An Agent builder"""
    # Build Agent
    try:
        llm = ChatOpenAI(
            temperature=st.session_state.temperature,
            model="gpt-3.5-turbo-16k",
            api_key=key,
        )
        agent = PDFExplainer(
            llm,
            client,
            extra_tools=st.session_state.wk_button,
        ).agent
    except Exception:  # pylint: disable=broad-exception-caught
        st.warning("Missing OpenAI API Key.")

    return agent
