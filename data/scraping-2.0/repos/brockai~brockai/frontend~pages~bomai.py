
import streamlit as st
import pandas as pd

from openai import OpenAI
st.set_page_config(layout="wide", page_title="brockai - BOM Compliancy", page_icon="./static/brockai.png")  

# import logging
# logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
# logging.getLogger("haystack").setLevel(logging.INFO)

from helpers.config import opensearch_platform, scheme, openaikey
from helpers.markdown import sidebar_links_footer, sidebar_app_header, powered_by_openai, platform_link
from services.api import upload

client = OpenAI(api_key=openaikey)   

if "messages_bom" not in st.session_state:
    st.session_state["messages_bom"] = [{"role": "assistant", "content": "Would you like to learn more about how to check your BOM for compliancy?"}]
    
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
    
st.header("💯 BOM Component Compliancy")
with open('styles.css') as f:
    st.markdown(
        f'<style>{f.read()}</style>'
        +powered_by_openai
        , unsafe_allow_html=True
    )

with open('styles.css') as f:
    st.sidebar.markdown(
        f'<style>{f.read()}</style>'
        +sidebar_app_header
        +sidebar_links_footer
        , unsafe_allow_html=True
    )
    
st.sidebar.markdown(platform_link, unsafe_allow_html=True)

uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    # st.write(bytes_data)

if st.button("🚀 Upload & Process", disabled=not uploaded_files):
  const 
  upload()
    
 
for msg in st.session_state.messages_bom:
    if msg["role"] == 'assistant':
      st.chat_message(msg["role"],avatar="🕵️‍♀️").write(msg["content"])
    else:
      st.chat_message(msg["role"]).write(msg["content"])
    
if prompt := st.chat_input():
    if not openaikey:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    st.session_state.messages_bom.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    response = client.chat.completions.create(model=st.session_state.openai_model, messages=st.session_state.messages_bom)
    msg = response.choices[0].message.content
    
    st.session_state.messages_bom.append({"role": "assistant", "content": msg})
    st.chat_message("assistant",avatar="🕵️‍♀️").write(msg)