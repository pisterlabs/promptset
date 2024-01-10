import streamlit as st
from openai import OpenAI
from helper import api_key
from main import parse_document
import json


# Initialize the session key for the text. See the end of parse_document() for writing.
if "text" not in st.session_state:
    st.session_state["text"] = ""
else:
    text = st.session_state["text"]


def main():
    # Current page sidebar
    st.sidebar.title("Chat Mode")
    st.sidebar.markdown("""
    Use this tab to get answers about your document.\n
    TODO: 
    - [ ] Fix "Clear All" button. Cache is not cleared.
    """)

    # Top level greeting

    st.title("Chat Mode")
    st.markdown("Get answers to your questions about your document.")
    st.header(' ') # Add some space


if __name__ == "__main__":
    main()

doc_loaded = st.empty()
if len(st.session_state["text"]) > 0:
    doc_loaded.info("Using document loaded in memory. You can also upload a new document below.")

# Upload file
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx"], help="Accepts PDF and Word documents.", key="chat_upload")
parsed_text, tokens, model = parse_document(uploaded_file)
if uploaded_file is not None:
    st.session_state["text"] = parsed_text
    text = parsed_text
    doc_loaded.info("Loading complete!")
else:
    text = st.session_state["text"]

# Request parameters
gen_max_tokens = 500
engine = "gpt-3.5-turbo-1106"
with open("userinfo.json", "r") as f:
    userinfo = json.load(f)
    if userinfo["install_flag"] == 1:
        endpoint = userinfo['endpoint'] # Use the local model if Local Mode is enabled

# Create the OpenAI request
client = OpenAI(api_key=api_key, base_url=endpoint)
sys_prompt = ("You are an assistant designed to give summaries of uploaded documents. Your answers should be decently long, "
              "in the form of bullet points. Make sure to include every point discussed in the document. Being verbose is "
              "highly preferable compared to missing ideas in the document. Here is the document to recap:")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What is your question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

# Initialize the full response variable
full_response = ""

# Load the document into the chat history
full_doc_prompt = (f"The document you need to answer questions about is:\n{text}\n\n"
                   f"Acknowledge the reception of the document and wait for user input to do anything.")

# Send the request to OpenAI
if st.session_state.messages:  # Check if 'messages' is not empty
    messages_to_send = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    # If there is text (document), prepend the full_doc_prompt for context without showing it in the UI
    if len(text) > 0:
        messages_to_send.insert(0, {"role": "system", "content": full_doc_prompt})

    # Create a chat message container for the assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for response in client.chat.completions.create(
                model=engine,
                messages=messages_to_send,
                stream=True,
                max_tokens=gen_max_tokens,
        ):
            full_response += (response.choices[0].delta.content or "")  # Handle empty or incoming response
            message_placeholder.markdown(full_response + "▌")  # Add a pipe to the end of the message to indicate typing since we're streaming
        message_placeholder.markdown(full_response)

    # Add the assistant message to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})



# Add a button to clear the chat history
def clear_chat_history():
    st.session_state.messages = []

def clear_all():
    st.session_state.messages = []
    st.session_state["text"] = ""
    del st.session_state["text"]
    st.stop()

if len(st.session_state.messages) > 0:
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
st.sidebar.button('Clear All', on_click=clear_all)


# ------------------- LICENSE -------------------
# Docuchat, a smart knowledge assistant for your documents.
# Copyright © 2024 xTellarin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/.
