import os

import openai
import streamlit as st
from dotenv import load_dotenv

from chains_bot import get_answer, get_answer_from_agent, run_multiple_agents
from supabase_db import getSimilarDocuments

load_dotenv()

st.title("GEN-TS HR Partner")

openai.api_key = os.getenv("OPENAI_API_KEY", "default_key")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = get_answer(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        