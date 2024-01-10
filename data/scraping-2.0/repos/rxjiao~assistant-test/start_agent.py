import openai
import time
import os
import sys
from dotenv import load_dotenv
from datetime import datetime

# Loading the .env file
load_dotenv()

# Getting the API key from the .env file
openai.api_key = os.getenv('OPENAI_API_KEY')

client = openai.OpenAI()

my_assistant = client.beta.assistants.retrieve(os.getenv('ASST_ID'))
print("\nRetrieved Assistant:\n", my_assistant.id, "\n")

# Log current time and assistant.id
with open('Conversation_logs.md', 'a') as f:
    f.write(f"## Assistant: {my_assistant.id}\n\n**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Read the last thread ID from a file
try:
    with open('last_thread_id.txt', 'r') as f:
        last_thread_id = f.read().strip()
except FileNotFoundError:
    last_thread_id = None

# If a last thread ID exists, ask the user whether to use it
if last_thread_id is not None:
    use_last_thread = input(f"A previous thread ID ({last_thread_id}) was found. Do you want to use it? (y/n, default is y): ")
    use_last_thread = use_last_thread.lower() or 'y'
    if use_last_thread == 'y':
        thread_id = last_thread_id
    else:
        # Delete the old thread
        response = client.beta.threads.delete(last_thread_id)
        print(response)

        # Initialize a new thread
        thread = client.beta.threads.create(
            messages=[]
        )
        thread_id = thread.id
else:
    # Initialize a new thread
    thread = client.beta.threads.create(
        messages=[]
    )
    thread_id = thread.id

# Write the new thread ID to a file
with open('last_thread_id.txt', 'w') as f:
    f.write(thread_id)

# Write Thread ID to conversation log file
with open('Conversation_logs.md', 'a') as f:
    f.write(f"**Thread**: {thread_id}\n\n")

print(f"\nConversation Now Starts. Type 'exit' to exit.\n")

while True:
    # Get user input
    user_input = input("\nUser: ")
    if user_input.lower() == 'exit':
        break

    # Add user message to the thread
    client.beta.threads.messages.create(
        thread_id,
        role="user",
        content=user_input
    )

    # Run the assistant on the thread
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=my_assistant.id
    )

    # Retrieve the run status
    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run.id
    )

    print("Thinking...", end='\r')
    sys.stdout.flush()  # Ensure "Thinking..." is printed immediately

    # Keep retrieving the run until its status is 'completed'
    while run.status != 'completed':
        time.sleep(1)  # Wait for a short period of time to avoid excessive requests
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )

    # Overwrite "Thinking..." with an empty string
    print(' ' * len("Thinking..."), end='\r')

    # Get the latest messages from the thread
    messages = client.beta.threads.messages.list(
        thread_id=thread_id
    )

    # Print the latest assistant message
    assistant_message = messages.data[0].content[0].text.value
    print("Assistant:", assistant_message, "\n")

    # Log this conversation
    with open('Conversation_logs.md', 'a') as f:
        f.write(f"**User**: {user_input}\n**Assistant**: {assistant_message}\n\n")
