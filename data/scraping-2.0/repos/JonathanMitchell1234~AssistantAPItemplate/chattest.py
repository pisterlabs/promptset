from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv()  

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)


# Replace with your actual Assistant ID
assistant_id = "asst_LvM5fVQc4nwVisQwCo3NUCVO"

# Your prompt to the Assistant
prompt = input('Enter your prompt: ')

# Create a Thread
thread = client.beta.threads.create()

# Add your prompt as a Message to the Thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=prompt
)

# Create a Run to get the Assistant's response
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant_id
)

# Wait for the run to complete
while True:
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    print(f"Run status: {run.status}")  # Print the run status
    if run.status == "completed":
        break
    time.sleep(1)  # Sleep for a short period before checking again

# Retrieve Messages from the Thread
messages = client.beta.threads.messages.list(
    thread_id=thread.id,
    order="asc"
)

# Print the Assistant's response
for message in messages:
    if message.role == "assistant":
        for text_or_image in message.content:
            if text_or_image.type == 'text':
                print(text_or_image.text.value)
               