import os

import openai
import streamlit as st


def chat_input():
    st.session_state.messages.append({
        "role": "user",
        "content": st.session_state.user_input
    })

    response = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=os.environ.get("OPENAI_MAX_TOKENS") or 1000,
        messages=st.session_state.messages,
    )

    st.session_state.messages.append(response["choices"][0]["message"])


# render history
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

with st.form(key="chat_input"):
    text_input = st.text_input("Message", key="user_input")
    submit_button = st.form_submit_button(label="Submit", on_click=chat_input)
