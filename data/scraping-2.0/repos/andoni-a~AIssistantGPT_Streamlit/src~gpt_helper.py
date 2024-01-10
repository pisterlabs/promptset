import openai
import json
import os

# Set your API key here
api_key = # Replace this with your OPENAI API key
openai.api_key = api_key
client = openai
assistant_id= # Replace this with your actual Assistant ID

assistant=client.beta.assistants.retrieve(assistant_id)

def initialize_client():
    """
    Initialize the OpenAI client and retrieve the assistant.
    """
    client = openai
    assistant = client.beta.assistants.retrieve(assistant_id)
    return client, assistant.id

def create_thread(client):
    """
    Create a new thread for a user conversation.
    """
    thread= client.beta.threads.create()
    return thread.id

def add_message_to_thread(client, thread_id, user_message):
    """
    Add a user's message to the specified thread.
    """
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )
    return message.id

def run_conversation(client, thread_id, assistant_id):
    """
    Run the conversation through the assistant and wait for completion.
    """
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=""
    )

    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        if run.status == "completed":
            break

    return run.id

def get_conversation_messages(client, thread_id):
    """
    Retrieve all messages from a conversation thread.
    """
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    message_json = messages.model_dump()
    for msg in messages.data:
        return (msg.content[0].text.value)

def get_run_steps(client, thread_id, run_id):
    """
    Retrieve the steps of a run.
    """
    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread_id,
        run_id=run_id
    )
    run_steps_json = run_steps.model_dump()
    return run_steps_json

def save_conversation_to_file(data, filename):
    """
    Save conversation data to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

