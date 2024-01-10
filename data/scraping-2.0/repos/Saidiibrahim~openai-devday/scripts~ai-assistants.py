from pathlib import Path
from openai import OpenAI
import io
from pydub import AudioSegment
from pydub.playback import play
import os
from dotenv import load_dotenv

# env variables
load_dotenv()
my_key = os.getenv('OPENAI_API_KEY')

# OpenAI API
client = OpenAI(api_key=my_key)

# create the assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview"
)

# create a thread
thread = client.beta.threads.create()

# add a message to a thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
)

# print thread messages
print("Thread messages:")
thread_messages = client.beta.threads.messages.list(message.thread_id)
print(thread_messages.data)
print("End of thread messages")

# Run the assistant
print("Running assistant...")
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="Please address the user as Jane Doe. The user has a premium account."
)

# Display the assistant's response
print("Assistant has responded:")
run = client.beta.threads.runs.retrieve(
  thread_id=thread.id,
  run_id=run.id
)

# retrieve the Messages added by the Assistant to the Thread
print("Assistant's messages:")
messages = client.beta.threads.messages.list(
  thread_id=thread.id
)

# print the messages
print(messages)