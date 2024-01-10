#!/usr/bin/env python3
import time
from openai import OpenAI

def load_api_key(filepath):
    with open(filepath, 'r') as file:
        return file.read().strip()

def main():
    # Load OpenAI API key from config file
    api_key = load_api_key('config.txt')
    if not api_key:
        print("API key not found in config file.")
        return

    client = OpenAI(api_key=api_key)

    # Load AI Assistants Exist
    assistants = client.beta.assistants.list(
        order="desc",
        limit="20",
    )

    # print("assistants.data is ", assistants.data)

    for k, v in enumerate(assistants.data):
        print(f"Index {k}, name {v.name}, desc {v.description}.")
    

    index = int(input("Please choose the AI you need: "))

    a_id = assistants.data[index].id

    thread = client.beta.threads.create()

    # Welcome message
    print("Welcome to the OpenAI Assistant. Type 'quit' to exit.")

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        message = client.beta.threads.messages.create(
            thread_id = thread.id,
            role="user",
            content=user_input,
        )

        run = client.beta.threads.runs.create(
            thread_id = thread.id,
            assistant_id = a_id,
        )

        while True:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

            # Check if the run status is 'completed'
            if run.status == 'completed':
                thread_messages = client.beta.threads.messages.list(thread.id)
                # print(thread_messages)

                if thread_messages:
                    print(f"Ans: {thread_messages.data[0].content[0].text.value}")
                break

            time.sleep(1)


if __name__ == "__main__":
    main()
