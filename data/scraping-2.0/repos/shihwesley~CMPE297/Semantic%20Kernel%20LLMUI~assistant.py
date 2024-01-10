from openai import OpenAI
import os
import json
import time

# Pretty printing helper
def pretty_print(messages):
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run

def display(obj):
    print(obj)

def show_json(obj):
    display(json.loads(obj.model_dump_json()))

openai_api_key = os.environ.get('OPENAI_API_KEY')

client = OpenAI(api_key= openai_api_key)

assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
    model="gpt-4-1106-preview",
)
show_json(assistant)

MATH_ASSISTANT_ID = assistant.id  # or a hard-coded ID like "asst-..."

def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")

def create_thread_and_run(user_input):
    thread = client.beta.threads.create()
    run = submit_message(MATH_ASSISTANT_ID, thread, user_input)
    return thread, run

# # Emulating concurrent user requests
# thread1, run1 = create_thread_and_run(
#     "I need to solve the equation `3x + 11 = 14`. Can you help me?"
# )
# thread2, run2 = create_thread_and_run("Could you explain linear algebra to me?")


# # Wait for Run 1
# run1 = wait_on_run(run1, thread1)
# pretty_print(get_response(thread1))

# # Wait for Run 2
# run2 = wait_on_run(run2, thread2)
# pretty_print(get_response(thread2))

while True:
    # Get user input
    user_input = input("Ask your math question: ")

    # Create thread and run
    thread, run = create_thread_and_run(user_input)

    # Wait for run to complete
    run = wait_on_run(run, thread)

    # Display responses
    pretty_print(get_response(thread))

    # Ask if user wants to continue
    cont = input("Do you want to ask another question? (y/n): ")
    if cont.lower() != "y":
        break


