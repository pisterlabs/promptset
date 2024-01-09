# Integrate to our Code for OpenAI
import os
from constant import openai_key
from langchain.llms import OpenAI

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key
st.title("Langchain Demo with OpenAI API")
input_text = st.text_input("Search the topic you Want")

## OPENAI LLMS
llm = OpenAI(temperature=0.8)  

if input_text:
    st.write(llm(input_text))
