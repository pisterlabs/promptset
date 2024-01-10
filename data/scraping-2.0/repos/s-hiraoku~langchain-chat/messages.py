import streamlit as st
from langchain.schema import SystemMessage


def init_messages():
    st.sidebar.title("Conversation")
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []
