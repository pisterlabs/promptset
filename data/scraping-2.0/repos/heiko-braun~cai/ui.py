# Import necessary libraries
import openai
import streamlit as st
import time

from conf.constants import *
from assistant import *

from langchain.schema import SystemMessage, AIMessage, HumanMessage

# Initialize session state variables for file IDs and chat control
if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Set up the Streamlit page with a title and icon
st.set_page_config(page_title="Camel Docs Assistants API", page_icon=":speech_balloon:")

# Create a sidebar for API key configuration and additional features
#st.sidebar.header("Configuration")
# api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
# if api_key:
#     openai.api_key = api_key

# Button to clear the output
# if st.sidebar.button("Clear Screen"):    
#     st.empty()

# st.markdown('''
# <style>
# .stApp [data-testid="stToolbar"]{
#     display:none;
# }
# </style>
# ''', unsafe_allow_html=True)

# Main chat interface setup
st.title("Camel Support Assistant")
assistant = Assistant(st)

# Only show the chat interface if the chat has been started

# Initialize the model and messages list if not already in session state
starter_message = "How can I help you?"
if "messages" not in st.session_state or st.button("Clear Thread"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]

# Display existing messages in the chat
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

if prompt := st.chat_input("How can I help you?"):
    # Add user message to the state and display it
    st.session_state.messages.append({"role": "user", "content": prompt.encode('iso-8859-1')})
    with st.chat_message("user"):
        st.markdown(prompt)

    assistant.kickoff(prompt)    
        