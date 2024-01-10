import time
from openai import OpenAI

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI()

assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview"
)

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
)

run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="Please address the user as Jane Doe. The user has a premium account."
)
# This creates a Run in a queued status. 
run = client.beta.threads.runs.retrieve(
  thread_id=thread.id,
  run_id=run.id
)


# periodically retrieve the Run to check on its status to see if it has moved to completed.

while run.status != "completed":
  run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
  )
  # sleep for 3 seconds
  time.sleep(3)

messages = client.beta.threads.messages.list(
  thread_id=thread.id
)

print(messages)