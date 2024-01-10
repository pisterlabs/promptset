import streamlit as st
import config.pagesetup as ps
import config.login as lg
from openai import OpenAI
import PyPDF2
import io
import docx
from utils.st.st_read_uploaded_file import read_file


# Setup: Set Variables, Constants, and Other Setup
client = OpenAI(api_key=st.secrets.openai.api_key_general)

#Session State
if "messages2" not in st.session_state:
    st.session_state.messages2 = []

# Page Title: Set Page Title
page_title = "FlowGenius AI"
page_subtitle = "File QA"
displayed_page_title = ps.set_title(page_title, page_subtitle)


# Define Functions: Define any functions used in this module
uploaded_file = st.file_uploader(
    label="Upload File", 
    type=["txt", "md", "docx", "pdf"]
)

question = st.text_input(
    label="Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file
)

if uploaded_file and question:
    toast_start = st.toast("FlowGenius AI is processing...", icon="🔨")
    file_content = read_file(uploaded_file)
    if file_content:
        prompt = f"""Provided is an article and a question from a user. Your job is to answer the question fully.
        {file_content}\n\n\n\n{question}"""
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages2.append(user_message)
        response = client.chat.completions.create(
            model=st.session_state.openai.model_gpt_turbo_normal_nofunctions,
            messages = st.session_state.messages2
        )
        response_msg = response.choices[0].message
        st.session_state.messages2.append(response_msg)
        response_content = response_msg.content
        toast_end = st.toast("FlowGenius AI is complete!", icon="✅")
        st.chat_message("assistant").markdown(response_content)