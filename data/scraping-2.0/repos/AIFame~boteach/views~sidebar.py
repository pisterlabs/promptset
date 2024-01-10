import openai
import streamlit as st


def sidebar():
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your OpenAI API key", type="password")
        orgId = st.text_input("Enter your OpenAI Organization ID [Optional]")
        if api_key:
            openai.api_key = api_key
            openai.organization = orgId
