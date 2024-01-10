import streamlit as st
from streamlit_openai import OpenAIConnection

st.title("ChatGPT AMA")
conn = st.experimental_connection("openai", type=OpenAIConnection)
input = st.text_input("Ask your question:")
if input:
    st.info(conn.get_completion(input))
