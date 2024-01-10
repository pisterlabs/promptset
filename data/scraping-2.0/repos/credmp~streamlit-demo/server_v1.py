#!/usr/bin/env python3

import os, streamlit as st

# os.environ['OPENAI_API_KEY']= ""

from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
)
from langchain.llms.openai import OpenAI

# Define a simple Streamlit app
st.title("Welcome to hackernotes")
query = st.text_input("What would you like to ask?", "")

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            print(query)
        except Exception as e:
            st.error(f"An error occurred: {e}")
