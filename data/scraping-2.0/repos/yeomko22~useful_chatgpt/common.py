import streamlit as st
import openai


openai.api_key = st.secrets["OPENAI_API_KEY"]


def write_page_config():
    st.set_page_config(
        page_title="AI 서비스 개발하기",
        page_icon="🧠"
    )


def request_chat_completion(prompt, stream=False, system_role=None):
    messages = [{"role": "user", "content": prompt}]
    if system_role:
        messages = [{"role": "system", "content": system_role}] + messages
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=stream
    )
    return response


def write_streaming_response(response):
    message = ""
    placeholder = st.empty()
    for chunk in response:
        delta = chunk.choices[0]["delta"]
        if "content" in delta:
            message += delta["content"]
            placeholder.markdown(message + "▌")
        else:
            break
    placeholder.markdown(message)
    return message
