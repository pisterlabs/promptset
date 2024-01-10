import openai
from openai import OpenAI
from config import openAiKey, assistant_id
import time

client = OpenAI(api_key=openAiKey)
# Create a new thread without an initial message
thread = client.beta.threads.create(messages=[])

while True:
    # Take user input
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Add the user's message to the thread
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input
    )

    # Create and run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
        model="gpt-4-1106-preview",
        instructions="You are a helpful assistant that understand what the user wants and based on that suggest shoes from the .json file, \
                      and also ask follow up questions to narrow down the list. Give top 5 shoes that matches the description",
        tools=[{"type": "code_interpreter"}, {"type": "retrieval"}]
    )

    # Wait for the run to complete
    run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id).status
    while run_status != 'completed':
        print("run_status = ", run_status)
        time.sleep(1)
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id).status


    # Retrieve and display the assistant's response
    messages = client.beta.threads.messages.list(thread_id=thread.id).data
    print("messages size = ", len(messages))
    for message in messages[0:2]:  # Get the latest message(s)
        if message.role == 'assistant':
            print("Assistant:", message.content[0].text.value)

