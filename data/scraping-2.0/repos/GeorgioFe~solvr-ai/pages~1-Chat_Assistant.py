'''
Author: Georgio Feghali
Date: July 11 2023
'''

# UI Dependencies.
import streamlit as st
from PIL import Image

# Logic Dependencies.
import openai

# Page Configuration.
favicon = Image.open("./admin/branding/logos/favicon-32x32.png")
st.set_page_config(
    page_title="Solvr.ai - Chat Assistant",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("<h1 style='text-align: center;'>Chat Assistant 🤖</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    openai.api_key = st.secrets['openai_api_key']
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)