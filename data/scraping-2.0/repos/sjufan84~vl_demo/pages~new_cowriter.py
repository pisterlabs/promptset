""" The new interface for co-writer incorporating the new "assistants API from openai """
from openai import OpenAI
import os
import sys
import json
from dotenv import load_dotenv
import streamlit as st
from pydantic import BaseModel, Field
from IPython.display import Audio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))))
from vl_demo.utils.assistant_utils import create_thread_run, add_message_and_run  

# Load environment variables
load_dotenv()

# Set up the openai client
api_key = os.getenv("OPENAI_KEY2")
organization = os.getenv("OPENAI_ORG2")

class ChatMessage(BaseModel):
    """ A chat message """
    role: str = Field(..., description="The role of the speaker")
    content: str = Field(..., description="The content of the message")

if "cowriter_messages" not in st.session_state:
    st.session_state.cowriter_messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

client = OpenAI(api_key=api_key, organization=organization, max_retries=3, timeout=10)
assistant_id = "asst_6Xf1yRhi5WG8rrQdCrqIPmaF"

def main():
  """ Main function for the chat page """
  if len(st.session_state.cowriter_messages) == 0:
    st.markdown(""" Welcome to Co-writer!  This session will simulate a very small taste of what
    we are working on.  By interacting with the artist, you can get feedback and suggestions on lyrics,
    and even generate a music clip to go along with your lyrics.  This is an early prototype, so the
    music generation may take a bit, but it will give you a flavor of the possibilities.  Enjoy!""")
  for message in st.session_state.cowriter_messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="🎸"):
            st.markdown(message["content"])
    elif message["role"] == "user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

  # Accept user input
  if prompt := st.chat_input("Hey friend, let's start writing!"):
      # Add user message to chat history
      st.session_state.cowriter_messages.append({"role": "user", "content": prompt})
      # Display user message in chat message container
      with st.chat_message("user"):
          st.markdown(prompt)
      # Load the prophet image for the avatar
      # Display assistant response in chat message container
      with st.spinner("Dave is thinking..."):
        if st.session_state.thread_id:
          response = add_message_and_run(thread_id=st.session_state.thread_id, message_content=prompt)
        else:
          response = create_thread_run(message_content=prompt)

        # Add assistant response to chat history
        st.session_state.cowriter_messages.append({"role": "assistant", "content": json.loads(response)["message"]})
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="🎸"):
            st.markdown(json.loads(response)["message"])
  if st.session_state.audio_bytes:
    with st.sidebar.container():
      st.markdown("**Current Audio Clip**")
      Audio(st.session_state.audio_bytes, autoplay=True)

if __name__ == "__main__":
  main()
