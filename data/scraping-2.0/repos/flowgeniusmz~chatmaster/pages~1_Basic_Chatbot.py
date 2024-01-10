import streamlit as st
import config.pagesetup as ps
import config.login as lg
from openai import OpenAI

# Setup: Set Variables, Constants, and Other Setup
client = OpenAI(api_key=st.secrets.openai.api_key_general)

model = st.secrets.openai.model_gpt_turbo_normal_nofunctions

init_asst_msg = {"role": "assistant", "content": "Welcome to FlowGeniusAI. How may I help you?"}

# Session State: Initialize session state
if "messages1" not in st.session_state:
    st.session_state.messages1 = []
    st.session_state.messages1.append(init_asst_msg)

if "messages1_user" not in st.session_state:
    st.session_state.messages1_user = []

if "messages1_assistant" not in st.session_state:
    st.session_state.messages1_assistant = []
    st.session_state.messages1_assistant.append(init_asst_msg)


# Page Title: Set Page Title
page_title = "FlowGenius AI"
page_subtitle = "Basic Chatbot"
displayed_page_title = ps.set_title(page_title, page_subtitle)


# Display messages
for msg in st.session_state.messages1:
    role = msg['role']
    
    content = msg['content']

    st.chat_message(role).markdown(content)
    #with st.chat_message(role):
        #st.markdown(content)
    

# Get User Input
if prompt := st.chat_input():

    user_msg = {"role": "user", "content": prompt}

    st.session_state.messages1.append(user_msg)

    response = client.chat.completions.create(
        model=model,
        messages=st.session_state.messages1
    )
    
    response_msg = response.choices[0].message

    st.session_state.messages1.append(response_msg)

    response_content = response_msg.content

    st.chat_message("assistant").markdown(response_content)