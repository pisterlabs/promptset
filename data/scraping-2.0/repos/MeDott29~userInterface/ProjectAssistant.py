from openai import OpenAI
from projectFunctions import ProjectAssistant as ProjectManager

client = OpenAI("your-api-key")

# Create an instance of ProjectManager
project_manager = ProjectManager()

# Define the functions
def create_project(name, description):
    return project_manager.create_project(name, description)

def list_projects():
    return project_manager.list_projects()

def view_project(name):
    return project_manager.view_project(name)

def update_project(name, description):
    return project_manager.update_project(name, description)

def delete_project(name):
    return project_manager.delete_project(name)

# Step 1: Create an Assistant
assistant = client.beta.assistants.create(
    name="Project Assistant",
    instructions="You use tools and functions to help manage and answer questions about our dashboard project.",
    tools=[
        {"type": "function", "function": {"name": "create_project", "description": "Create a new project"}},
        {"type": "function", "function": {"name": "list_projects", "description": "List all projects"}},
        {"type": "function", "function": {"name": "view_project", "description": "View details about a project"}},
        {"type": "function", "function": {"name": "update_project", "description": "Update a project"}},
        {"type": "function", "function": {"name": "delete_project", "description": "Delete a project"}}
    ],
    model="gpt-4-1106-preview"
)

# Step 2: Create a Thread
thread = client.beta.threads.create()

# Step 3: Add a Message to a Thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to know the status of the dashboard project. Can you help me?"
)

# Step 4: Run the Assistant
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="Please provide the status of the dashboard project."
)

# Step 5: Check the Run status
run = client.beta.threads.runs.retrieve(
  thread_id=thread.id,
  run_id=run.id
)

# Step 6: Display the Assistant's Response
messages = client.beta.threads.messages.list(
  thread_id=thread.id
)

for message in messages:
    print(f"{message.role}: {message.content}")
