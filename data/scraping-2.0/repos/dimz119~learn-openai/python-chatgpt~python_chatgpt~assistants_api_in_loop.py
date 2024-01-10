import time
from openai import OpenAI

# make sure you have OPENAI_API_KEY environment variable with API key
# export OPENAI_API_KEY=""

client = OpenAI()

# create an assistant
assistant = client.beta.assistants.create(
                name="Math Tutor",
                instructions="You are a personal math tutor. Write and run code to answer math questions.",
                tools=[{"type": "code_interpreter"}],
                model="gpt-4-1106-preview")

# create a thread. This will have unlimited messages, but won't be used all
thread = client.beta.threads.create()

def run_chat(msg: str) -> None:
    # create a message
    message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=msg)

    run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="Please address the user as Jane Doe. The user has a premium account.")

    while True:
        # queued, in_progress, completed
        if run.status != 'completed':
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id)
            print(f"{run.status} now ... waiting 3 secs ... ")
            time.sleep(3)
        else:
            break

    print(f"It is ready ....")

    messages = client.beta.threads.messages.list(
                thread_id=thread.id,
                order="asc",
                after=message.id)

    for thread_message in messages.data:
        print(thread_message.content[0].text.value)

stop_word = 'quit'

print(f"Thank you for choosing Superduper math tutor!! Type '{stop_word}' if you want to stop.")

while True:
    message = input("Question? ")
    if message == stop_word:
        break
    # I need to solve the equation `3x + 11 = 14`. Can you help me?
    run_chat(msg=message)

# clean up the assistant
client.beta.assistants.delete(assistant.id)
