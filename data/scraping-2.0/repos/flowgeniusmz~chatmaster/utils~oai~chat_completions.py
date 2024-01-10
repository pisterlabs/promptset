import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets.openai.api_key_general)
model = st.secrets.openai.model_gpt_turbo_normal_nofunctions

def create_chat_completion():

    response = client.chat.completions.create(
        model=model,
        messages=st.session_state.message1
    )

    response_message = response.choices[0].message
    
    st.session_state.messages1.append(response_message)

    response_content = response_message.content

    return response_content
