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

assistant = client.beta.assistants.create(
    name="Coder",
    instructions="You are a professional Python programmer.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview",
)

MATH_ASSISTANT_ID = assistant.id


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
    run = submit_message(MATH_ASSISTANT_ID, thread, user_input)
    return thread, run


thread, run = create_thread_and_run(
    "Generate the first 20 fibbonaci numbers with code."
)


def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")


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


run = wait_on_run(run, thread)
pretty_print(get_response(thread))

run_steps = client.beta.threads.runs.steps.list(
    thread_id=thread.id, run_id=run.id, order="asc"
)

for step in run_steps.data:
    step_details = step.step_details
    print(json.dumps(show_json(step_details), indent=4))