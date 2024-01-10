# Importing required packages
import streamlit as st
import openai
import uuid
import time

from openai import OpenAI

client = OpenAI()

MODEL = "gpt-4-1106-preview"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "run" not in st.session_state:
    st.session_state.run = {"status": None}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retry_error" not in st.session_state:
    st.session_state.retry_error = 0

st.set_page_config(page_title="BeeBoop: a Beeswax Chatbot")
st.sidebar.title("Ask me anything!")
st.sidebar.divider()
st.sidebar.markdown("Current Version: 0.0.3")
st.sidebar.markdown("Using gpt-4-1106-preview API")
st.sidebar.markdown(st.session_state.session_id)
st.sidebar.divider()

if "assistant" not in st.session_state:
    openai.api_key = st.secrets["OPENAI_API_KEY"]

    # Load the previously created assistant
    st.session_state.assistant = openai.beta.assistants.retrieve(
        st.secrets["OPENAI_ASSISTANT"]
    )

    # Create a new  for this session
    st.session_state.thread = client.beta.threads.create(
        metadata={
            "session_id": st.session_state.session_id,
        }
    )

# If the run is completed, display the messages
elif (
    hasattr(st.session_state.run, "status")
    and st.session_state.run.status == "completed"
):
    # Retrieve the list of messages
    st.session_state.messages = client.beta.threads.messages.list(
        thread_id=st.session_state.thread.id
    )

    # Display messages
    for message in reversed(st.session_state.messages.data):
        if message.role in ["user", "assistant"]:
            with st.chat_message(message.role):
                for content_part in message.content:
                    message_text = content_part.text.value
                    st.markdown(message_text)

if prompt := st.chat_input("How can I help you?"):
    with st.chat_message("user"):
        st.write(prompt)
    
    # Add message to the thread
    st.session_state.messages = client.beta.threads.messages.create(
        thread_id=st.session_state.thread.id, role="user", content=f" Use the provided documents as context to answer this question: {prompt}"
    )

    # Do a run to process the messages in the thread
    st.session_state.run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread.id,
        assistant_id=st.session_state.assistant.id,
    )
    if st.session_state.retry_error < 3:
        time.sleep(1)  # Wait 1 second before checking run status
        st.rerun()

# Check if 'run' object has 'status' attribute
if hasattr(st.session_state.run, "status"):
    # Handle the 'running' status
    if st.session_state.run.status == "running":
        with st.chat_message("assistant"):
            st.write("Thinking ......")
        if st.session_state.retry_error < 3:
            time.sleep(1)  # Short delay to prevent immediate rerun, adjust as needed
            st.rerun()

    # Handle the 'failed' status
    elif st.session_state.run.status == "failed":
        st.session_state.retry_error += 1
        with st.chat_message("assistant"):
            if st.session_state.retry_error < 3:
                st.write("Run failed, retrying ......")
                time.sleep(3)  # Longer delay before retrying
                st.rerun()
            else:
                st.error(
                    "FAILED: The OpenAI API is currently processing too many requests. Please try again later ......"
                )

    # Handle any status that is not 'completed'
    elif st.session_state.run.status != "completed":
        # Attempt to retrieve the run again, possibly redundant if there's no other status but 'running' or 'failed'
        st.session_state.run = client.beta.threads.runs.retrieve(
            thread_id=st.session_state.thread.id,
            run_id=st.session_state.run.id,
        )
        if st.session_state.retry_error < 3:
            time.sleep(3)
            st.rerun()
