import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def load_chat_model():
    load_dotenv()
    openai_api_key = st.session_state.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    temperature = st.session_state['temperature']
    max_tokens = st.session_state['max_tokens']
    model_name = st.session_state.get("selected_model_option")

    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=temperature, max_tokens=max_tokens, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    return llm
