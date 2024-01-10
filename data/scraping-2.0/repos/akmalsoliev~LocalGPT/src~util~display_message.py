import os 
import streamlit as st 
from langchain.schema import (
    AIMessage,
    HumanMessage
)

def display_messages():
    for index, message in enumerate(st.session_state.messages):
        if type(message) in [AIMessage]:
            with st.chat_message("assistant"):
                st.write(message.content)

        elif type(message) == HumanMessage:
            with st.chat_message("user"):
                st.write(message.content)
