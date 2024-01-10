# Imports

from openai import OpenAI
import dotenv
import os
from time import sleep
import json
from types import SimpleNamespace

dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ask if previous run should be used.

use_run = input("Use previous run? (y/n): ")

# Yes? Load the run from the json file.

if use_run == "y":
    with open("run.json") as json_file:
        data = json.load(json_file)

    assistant = SimpleNamespace(id=data["assistant_id"])
    thread = SimpleNamespace(id=data["thread_id"])
    run = SimpleNamespace(id=data["run_id"])

# No? Create a new assistant, thread and run.

if use_run == "n":

    # create the assistant

    assistant = client.beta.assistants.create(
        name="Math tutor",
        instructions="You are a helpfull math instructor. Write and run code to answer math questions.",
        tools=[{
            "type": "code_interpreter"
        }],
        model="gpt-3.5-turbo-1106"
    )

    # Create the thread

    thread = client.beta.threads.create()

while True:

    user_choice = input("Do you want to ask a question? (y/n): ")

    if user_choice == "n":
        break

    # Get user question input

    user_question = input("What is your math question? ")

    # Add message to thread

    message = client.beta.threads.messages.create(
        thread_id = thread.id,
        role= "user",
        # content="Solve this problem: 3x + 11 = 14"
        content=user_question
    )

    # Create the run

    run = client.beta.threads.runs.create(
        thread_id = thread.id,
        assistant_id = assistant.id,
    )

    # Save the assistant, thread and run to a json file for later use.

    data = {
        "assistant_id": assistant.id,
        "thread_id": thread.id,
        "run_id": run.id
    }

    with open("run.json", "w") as json_file:
        json.dump(data, json_file)

    # Retrieve information about the run and wait for it to complete.

    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )

    while run.status != "completed":
        sleep(2)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

        if run.status == "failed":
            print("Failed")
            break

    # Find the code interpreter step and print the output.

    # obs.: Should have a better way to do this, verifying if the step is a code interpreter step. And if it has an output.

    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread.id,
        run_id=run.id
    )

    interpreter_code = run_steps.data[1].step_details.tool_calls[0].code_interpreter.input

    # Get the list of messages between assistant and user.

    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )


    print("\n")

    # Print the messages

    for message in reversed(messages.data):
        print(message.role + ": "+message.content[0].text.value)
        print("\n")

    # Print with code from code interpreter
    print("\n\n")
    print("Interpreter code: " + interpreter_code)

