import json
import openai
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


def show_json(obj):
    print(json.loads(obj.model_dump_json()))


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Upload the file
file = client.files.create(
    file=open(
        "docs/2023q3-alphabet-earnings-release.pdf",
        "rb",
    ),
    purpose="assistants",
)
# create Assistant
assistant = client.beta.assistants.create(
    name="Coder",
    instructions="You are an expert in extracting information from documents.",
    tools=[{"type": "code_interpreter"}, {"type": "retrieval"}],
    model="gpt-4-1106-preview",
    file_ids=[file.id],
)

ASSISTANT_ID = assistant.id


def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )


def create_thread_and_run(user_input):
    thread = client.beta.threads.create()
    run = submit_message(ASSISTANT_ID, thread, user_input)
    return thread, run


thread, run = create_thread_and_run(
    "What is Deepmind working on?"
)


def pretty_print(messages):
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()


# Waiting in a loop
def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")


run = wait_on_run(run, thread)
pretty_print(get_response(thread))
