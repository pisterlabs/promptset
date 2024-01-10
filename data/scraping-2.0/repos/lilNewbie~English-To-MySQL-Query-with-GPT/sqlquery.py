from urllib import response
import streamlit as st
import random 
import time
import openai
from openai import OpenAI
import requests

st.title("Aloo_GPT")

client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
GPT_MODEL = 'gpt-3.5-turbo-0613'

def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    
    json_data = {"message": messages[0]["content"]}
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})
    try:
        response = requests.post(
            "http://localhost:5000/get_response",
            json=json_data,
        )
        return response.text
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


if 'openai_model' not in st.session_state:
    st.session_state['openai_model']="gpt-3.5-turbo"

if 'messages' not in st.session_state:
    st.session_state.messages=[]


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


if prompt := st.chat_input('Hello! What can I help you with?'):
    st.session_state.messages.append({'role':'user','content':prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        full_response = chat_completion_request([st.session_state.messages[-1]])
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({'role':'assistant','content':full_response})
