import os
#from dotenv import load_dotenv
import streamlit as st
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from vertexai.preview.generative_models import GenerativeModel
from tolkai_llm import chat_tolkai


class TOLKAILLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "tolkai"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        if chat_tolkai(prompt):
            return chat_tolkai(prompt)
        else:
            return "<No answer given by Gemini Pro>"


# Initialize Vertex AI
#load_dotenv()
#project_name = os.getenv("VERTEXAI_PROJECT")
vertexai.init(project="ping38", location="us-central1")


# Setting page title and header
st.set_page_config(page_title="TolkAI", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>TolkAI</h1>", unsafe_allow_html=True)


# Initialise session state variables
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.sidebar.title("Sidebar")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Reset conversation
if clear_button:
    st.session_state['messages'] = []

# Display previous messages
for message in st.session_state['messages']:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)

# Chat input
prompt = st.chat_input("You:")
if prompt:
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = chat_tolkai(prompt)
    st.session_state['messages'].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)