import openai
import os
import time
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

st.header("OpenAI Code Interpreter Assistant! ")

#  tools : Code Interpreter, Retrieval, and Function calling
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Step 1: Create an Assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview"
)

# Step 2: Create a thread
thread = client.beta.threads.create()

user_message = st.text_input("Pls enter your question:")
submit = st.button("submit", type="primary")

if submit:
    # Step 3: Add a Message to a Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message
    )

    # Step 4: Run the Assistant

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account."
    )

    # Step 5: Display the Assistant's Response
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )

    time.sleep(10)

    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )

    run_steps = client.beta.threads.runs.steps.list(
            thread_id=thread.id,
            run_id=run.id
        )

    if run_steps.data[0].status == "completed":
        print("The run was successful!")
        messages = client.beta.threads.messages.list(
                thread_id=thread.id
        )
        for message in messages.data:
            print(message.role.upper(), message.content[0].text.value)
            role = message.role.upper()
            content = message.content[0].text.value
            st.write(role, " : ", content)