import os
import openai
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


from ute import init_openai

(Client, LLM)=init_openai()

assistant = Client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview"
)

thread = Client.beta.threads.create()

message = Client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
)

run = Client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="Please address the user as Jane Doe. The user has a premium account."
)

print("RUN STARTED.  run id: ", run.id)

print(run.status, end=" ")
while run.status != "completed":
    run = Client.beta.threads.runs.retrieve(thread_id=run.thread_id, run_id=run.id)
    print(run.status, end=" ")
    if run.status == "completed":
        print("Completed")
        break
    else:
        time.sleep(1)

messages = Client.beta.threads.messages.list(thread_id=thread.id)
print(messages.data[0].content[0].text.value)

