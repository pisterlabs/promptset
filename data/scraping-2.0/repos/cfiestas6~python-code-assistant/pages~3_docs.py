import streamlit as st
from langchain.llms import Ollama

st.set_page_config(page_title="Docs Generation")
st.sidebar.header("Docs Generation")
st.header("Docs Generation")

if code := st.text_area(label="Paste your code", max_chars=None, height=400):
    prompt = f"DO NOT PROVIDE ANY INTRO. Generate shor documentation for this code. Provide a short description for each function. Code: {code}"
    llm = Ollama(model="codellama:13b-instruct")
    res = llm.predict(prompt)
    st.write(res)
