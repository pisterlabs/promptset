import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
import pinecone
from dotenv import load_dotenv


load_dotenv()
DEFAULT_SELECT_VALUE = "Select repository"
MODEL_NAME = "gpt-3.5-turbo-16k"


def initialize_session():
    if not "initialized" in st.session_state:
        st.session_state["initialized"] = True
        st.session_state["repo_name"] = DEFAULT_SELECT_VALUE
        st.session_state["user_name"] = ""
        st.session_state["repo_url"] = ""
        st.session_state["visitied_list"] = []
        st.session_state["messages"] = []
        st.session_state["chat_memory"] = None
        st.session_state["llm"] = ChatOpenAI(
            temperature = 0.0
        )
        pinecone.init(
            api_key = os.environ["PINECONE_API_KEY"],
            environment = os.environ["PINECONE_ENV"]
        )
        st.session_state["index_name"] = os.environ["PINECONE_INDEX_NAME"]


def handling_user_change():
    st.session_state["repo_name"] = DEFAULT_SELECT_VALUE
    st.session_state["repo_url"] = ""