import streamlit as st
from openai.error import OpenAIError

# custom imports
from splitText import split_text
from readFiles import read_pdf

def processPDFdata(uploaded_file):
    text = read_pdf(uploaded_file)
    with st.spinner("Splitting text into chunks ..."):
        chunks = split_text(text)
    # except OpenAIError as e:
    #     st.error(e._message)
    chunks = split_text(text)
    return chunks
