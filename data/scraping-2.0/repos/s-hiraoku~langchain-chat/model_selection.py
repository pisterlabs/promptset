import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from config import get_openai_api_key


def select_model():
    openai_api_key = get_openai_api_key()

    st.sidebar.title("Options")
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo-0613"
    elif model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo-16k-0613"
    else:
        st.session_state.model_name = "gpt-4"

    # 300: 本文以外の指示のtoken数 (以下同じ)
    st.session_state.max_token = (
        OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    )
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01
    )
    return ChatOpenAI(
        temperature=temperature,
        model_name=st.session_state.model_name,
        openai_api_key=openai_api_key,
        streaming=True,
    )
