import streamlit as st
import langchain_helper as lch
from dotenv import load_dotenv
import os

load_dotenv("openapi_key.txt")
openai_api_key = os.getenv("OPENAI_API_KEY")

st.title(":mag: Extract information from text")

uploaded_file = st.file_uploader("Upload a text file")

# Generate button
if st.button("Extract"):
    if uploaded_file is not None:
        response = lch.extract_info_from_text(
            openai_api_key=openai_api_key, file=uploaded_file
        )
        st.dataframe(response)
