# Python advanced example of how to build an interactive chat with the OpenAI Assistant API.

import os
import time
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Assistant ID (can be a hard-coded ID)
ASSISTANT_ID = 'YOUR_ASSISTANT_ID'

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Create an OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to submit a message to the assistant
def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=user_message)
    return client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)

# Function to get a response from the assistant
def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")

# Function to wait for the assistant's response with a spinner
def wait_on_run(run, thread):
    spinner = ['|', '/', '—', '\\']
    spinner_count = 0

    while run.status == "queued" or run.status == "in_progress":
        # Spinner visual
        sys.stdout.write("\r" + spinner[spinner_count % len(spinner)])
        sys.stdout.flush()
        spinner_count += 1

        # Check run status
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        time.sleep(0.1)

    # Clear spinner
    sys.stdout.write("\r \r")
    sys.stdout.flush()

    return run

# Pretty printing helper for the last exchange
def pretty_print(messages):
    if len(messages.data) >= 2:
        last_assistant_message = messages.data[-1]
        print(f"Assistant: {last_assistant_message.content[0].text.value}")

# Main chat loop
def chat_loop():
    assistant = client.beta.assistants.retrieve(ASSISTANT_ID)
    print(f"Welcome to the {assistant.name} Chat. Type 'exit' to quit.")
    thread = client.beta.threads.create()

    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            break

        run = submit_message(ASSISTANT_ID, thread, user_input)
        run = wait_on_run(run, thread)
        responses = get_response(thread)
        pretty_print(responses) 
      
chat_loop()
