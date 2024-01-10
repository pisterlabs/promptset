# if error: ImportError: cannot import name 'OpenAI' from 'openai'
# pip install --upgrade openai
# streamlit run gpt_assistant.py

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(

    api_key=os.getenv("OPENAI_API_KEY")
    # api_key=st.secrets["OPENAI_API_KEY"]
)

def get_session_state():
    session_state = st.session_state
    if 'first_time' not in session_state:
        session_state.first_time = True
        session_state.messages = []  # Initialize an empty list to store messages
        # assistant = client.beta.assistants.create(
        #     name='my-assistant',
        #     instructions="You are an AI assistant.",
        #     model='gpt-3.5-turbo-1106',
        #     tools=[{'type':'retrieval'}]
        # )
        # print(assistant.id)
    return session_state

session_state = get_session_state()

if session_state.first_time:
    thread = client.beta.threads.create()
    session_state.thread_id = thread.id
    # session_state.assistant_id = st.secrets["ASSISTANT_ID"]
    session_state.assistant_id = os.getenv("ASSISTANT_ID")
    session_state.first_time = False

user_input = st.text_input(label="Enter something:", placeholder="Type here...")

def create_message(query, thread_id):
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=query
    )

def bot_response(thread_id, assistant_id):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            latest_message = messages.data[0]
            text = latest_message.content[0].text.value
            break
    return text

if st.button("Submit"):
     with st.spinner("Processing"):
        user_message = user_input
        create_message(user_message, session_state.thread_id)
        bot_reply = bot_response(session_state.thread_id, session_state.assistant_id)

        # session_state.messages.insert(0, ("User", user_message))
        # session_state.messages.insert(0, ("Bot", bot_reply))
        
        session_state.messages.append(("User", user_message))
        session_state.messages.append(("Bot", bot_reply))

# Display all messages
for role, message in session_state.messages:
    st.write(f"{role}:", message)

# Add a reset button
if st.button("Reset Session"):
    st.session_state.clear()
    st.experimental_rerun()  # Rerun the app to reset the session