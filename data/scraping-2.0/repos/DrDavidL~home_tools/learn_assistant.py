from openai import OpenAI
import streamlit as st
from prompts import *
import random
import time
from typing import List, Optional, Union, Dict, Any

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


st.title("My Teacher!")

if check_password():

    client = OpenAI(
    organization= st.secrets["ORGANIZATION"],
    api_key = st.secrets["OPENAI_API_KEY"]
    )
    # Retrieve My Assistant
    my_assistant = client.beta.assistants.retrieve(st.secrets["ASSISTANT_ID"])

    # Create a new thread
    thread = client.beta.threads.create()

    # Add a message to the thread
    my_name = st.text_input("What is your name?")
    my_question = st.text_input("What is your question?")
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f'user_name: {my_name} Question: {my_question}'
    )

    # Run the assistant
    if st.button("Ask your question!"):
        my_run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=my_assistant.id,
            instructions=bio_tutor,
            )
        
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        # Periodically retrieve the Run to check on its status to see if it has moved to completed
        while my_run.status != "completed":
            keep_retrieving_run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=my_run.id
            )
            # st.write(f"Run status: {keep_retrieving_run.status}")

            if keep_retrieving_run.status == "completed":
                # print("\n")
                break
        all_messages = client.beta.threads.messages.list(
        thread_id=thread.id
        )

        with st.chat_message("user"):
            st.write(my_question)
            
        with st.chat_message("assistant"):
            st.write(all_messages.data[0].content[0].text.value)
