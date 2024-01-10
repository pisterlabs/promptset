import streamlit as st
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "..")

import os
import tempfile
import openai
from src.loader import ingest
from src.qa import bot


# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("📝 Chat with Your Data")

# Upload a file
uploaded_file = st.file_uploader("Upload a file", type=("txt", "md", "pdf"))

if uploaded_file:
    article = uploaded_file.read().decode()
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(article.encode())
        fp.seek(0)
        db = ingest(fp.name)
        st.success(f"Your file is successfully ingested!", icon="✅")

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask something about the file you uploaded..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in bot(db, st.session_state.messages, "gpt-3.5-turbo"):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
