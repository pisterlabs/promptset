"""Quickly create a Proof of Concept UI for your chatbot"""
import os
from dotenv import load_dotenv
import openai
import streamlit as st
from functions import Functions
from ui_helper import UIHelper

# Load OpenAI API key from your .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Use CSS to add custom styles to your UI
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Side bar
st.sidebar.title("Clear Chat")
if st.sidebar.button("Clear Chat", key="clear_chat"):
    st.session_state.messages = []
st.sidebar.title("Instructions")
st.sidebar.markdown("1. create a .env file with your OPENAI_API_KEY")
st.sidebar.markdown("2. install dependencies:\n pip install -r requirements.txt")
st.sidebar.markdown("3. in your terminal, run:\n streamlit run ChatPOC.py")

# Main content
st.title("ChatPOC")
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if UIHelper.is_valid_json(content):
            UIHelper.render_json(content)
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        is_function_json = False
        # Optional loading icon while waiting for response
        with st.spinner("✨🤖✨"):
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                functions=Functions().functions,
                function_call="auto",
                temperature=0,
                stream=True,
            ):
                stream_text = response.choices[0].delta
                if hasattr(stream_text, "function_call"):
                    is_function_json = True
                    full_response += stream_text.function_call.get("arguments", "")
                elif hasattr(stream_text, "content"):
                    full_response += stream_text.get("content", "")    
                message_placeholder.markdown(full_response + "▌")

            if is_function_json:
                message_placeholder.empty()
                UIHelper.render_json(full_response)
            else:
                message_placeholder.markdown(full_response)    
    st.session_state.messages.append({"role": "assistant", "content": full_response}) 
