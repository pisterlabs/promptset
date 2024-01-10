import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai_api_key"])

response = client.beta.assistants.delete("asst_koCCeFLh6H8aFU1VMq2muLRE")
print(response)