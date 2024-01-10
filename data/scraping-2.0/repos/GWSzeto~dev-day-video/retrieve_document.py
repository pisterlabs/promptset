from openai import OpenAI
from dotenv import load_dotenv
from time import sleep
import os
load_dotenv()

# Create an instance of OpenAI Client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Upload a file with an "assistants" purpose
file = client.files.create(
    file=open("attention_is_all_you_need.pdf", "rb"),
    purpose="assistants",
)

# Create an assistant
# Specializes in understanding and answering questions about research papers
assistant = client.beta.assistants.create(
    name="Research Assistant",
    instructions="You are a research assistant. Read and Write answers to research questions.",
    tools=[{"type": "retrieval"}],
    model="gpt-4-1106-preview",
    file_ids=[file.id],
)

# This is the channel that the user and assistant will communicate through
thread = client.beta.threads.create()

# User sends a message to the assistant
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Summarize the contents of the research paper 'Attention is all you need'"
)

# assistant processes the message sent by the user
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
)

# checks the status of the response from the assistant
while run.status != "completed":
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    sleep(2)

# # Shows Steps taken by the assistant to solve the math problem
# steps = client.beta.threads.runs.steps.list(
#     thread_id=thread.id,
#     run_id=run.id,
# )
# print(steps.model_dump_json())

# Shows the interaction of messages between the user and the assistant
messages = client.beta.threads.messages.list(
    thread_id=thread.id,
)
print(messages.model_dump_json())

