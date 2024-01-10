import openai
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

#  tools : Code Interpreter, Retrieval, and Function calling
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Upload a file with an "assistants" purpose
file = client.files.create(
    file=open("rag_assistant.py", "rb"),
    purpose='assistants'
)

# Create an assistant using the file ID
assistant = client.beta.assistants.create(
    instructions="You are an expert Python programmer. Write and run code to answer questions about Python.",
    model="gpt-4-1106-preview",
    tools=[{"type": "code_interpreter"}],
    file_ids=[file.id]
)

thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": "Can you write a unit test for the Python code please?",
            "file_ids": [file.id]
        }
    ]
)

run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Jane Doe. The user has a premium account."
)

time.sleep(20)

run_steps = client.beta.threads.runs.steps.list(
    thread_id=thread.id,
    run_id=run.id
)

if run_steps.data[0].status == "completed":
    print("The run was successful!")
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    for message in messages.data:
        role = message.role.upper()
        content = message.content[0].text.value
        print(f"{role}: {content}")