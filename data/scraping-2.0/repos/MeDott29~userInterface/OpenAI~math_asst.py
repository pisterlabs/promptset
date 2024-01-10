from openai import OpenAI

client = OpenAI("your-api-key")

# Define the function
def add_numbers(num1, num2):
    return num1 + num2

# Create an Assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "function", "function": {"name": "add_numbers", "description": "Add two numbers"}}],
    model="gpt-4-1106-preview"
)

# Create a Thread
thread = client.beta.threads.create()

# Add a Message to a Thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="add_numbers 5 7"
)

# Run the Assistant
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="Please add the numbers."
)

# Check the Run status
run = client.beta.threads.runs.retrieve(
  thread_id=thread.id,
  run_id=run.id
)

# Display the Assistant's Response
messages = client.beta.threads.messages.list(
  thread_id=thread.id
)

for message in messages:
    print(f"{message.role}: {message.content}")