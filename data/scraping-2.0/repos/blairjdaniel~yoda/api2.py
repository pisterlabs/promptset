from openai import OpenAI
import json
import time
from IPython.display import display

ASSISTANT_ID = "asst_iI7puRBKhk6QyA8TJfOg9V5v"  # or a hard-coded ID like "asst-..."

client = OpenAI()
user_input = ""
instruction = "Please address the user as Blair. The user has a free account"

def show_json(obj):
    display(json.loads(obj.model_dump_json()))

def add_message_and_run(thread_id, user_input):
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_input
    )
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID
    )
    return message, run

    run_run(run, thread_id)

def run_run(run, thread_id):
    run = wait_on_run(run, thread_id)

def get_response(thread_id):
    return client.beta.threads.messages.list(thread_id=thread_id, order="asc")

def wait_on_run(run, thread_id):
    while run.status == "queued" or run.status == "in_progress":
        time.sleep(0.5)  # This should be inside the loop to wait before each check
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id,
        )
    return run

# Create a new thread
thread = client.beta.threads.create()

# Add a message and start a run
message, run = add_message_and_run(thread.id, "Hello, assistant!")
wait_on_run(run, thread.id)

# Add another message and start another run
message, run = add_message_and_run(thread.id, "How are you?")
wait_on_run(run, thread.id)
