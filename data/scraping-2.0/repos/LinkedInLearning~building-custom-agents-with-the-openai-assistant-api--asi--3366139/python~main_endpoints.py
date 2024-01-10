# Python basic example of how to use the OpenAI API to create:
# - assistant
# - thread
# - message
# - run.
#
# Examples taken from https://cookbook.openai.com/examples/assistants_api_overview_python
# Full API documentation: https://platform.openai.com/docs/api-reference/assistants 

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load the environment variables from the .env file in the root.
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Create an OpenAI client
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv('OPENAI_API_KEY'),
)

# Create an assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
    model="gpt-4-1106-preview",
)
print(f"The assistant object:\n{assistant}\n")

# Create a thread
thread = client.beta.threads.create()
print(f"The thread object:\n{thread}\n")

# Create a message
message = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
)
print(f"The message object:\n{message}\n")

# Create a run
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
print(f"The run object:\n{run}\n")

# Wait for the run to complete
def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run
run = wait_on_run(run, thread)

# Get the response object
response = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
print(f"The response object:\n{response}\n")

# Print the assistant's response
print(f"Assistant: {response.data[-1].content[0].text.value}")
    