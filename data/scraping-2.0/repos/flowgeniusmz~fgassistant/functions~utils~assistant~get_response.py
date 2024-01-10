import streamlit as st
from openai import OpenAI


client = OpenAI(api_key = st.secrets.openai.api_key)

def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")
