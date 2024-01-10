import streamlit as st
from langchain.llms import Ollama

st.set_page_config(page_title="Code Completion")
st.sidebar.header("Code Completion")

st.header("Code Completion")
if code := st.text_area(label="Paste your code", max_chars=None, height=400):
    llm = Ollama(model="codellama:13b-python")
    res = llm.predict(code)
    result = str(code + res)
    st.code(result)
