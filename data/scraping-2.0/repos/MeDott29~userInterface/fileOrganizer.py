from openai import OpenAI
import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
directory_structure = list_files('/home/m/userInterface')
client = OpenAI()

assistant = client.beta.assistants.create(
    name="File Organizer",
    instructions="You are an assistant that helps users organize their files and directories. You can list, move, rename, delete, and create files and directories.",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List all files and directories in a given directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "The directory to list"}
                    },
                    "required": ["directory"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "move_file",
                "description": "Move a file from one directory to another",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "The file to move"},
                        "destination": {"type": "string", "description": "The directory to move the file to"}
                    },
                    "required": ["source", "destination"]
                }
            }
        },
        # Define additional functions here...
    ],
    model="gpt-3.5-turbo-1106"
)

# Step 2: Create a Thread
thread = client.beta.threads.create()

# Get the directory structure. This has to be abstracted out to fit the needs of multiple users

# Step 3: Add a Message to a Thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Provide suggestions on how I can improve the organization of my project."
)

# Step 4: Run the Assistant
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="You are an assistant that helps users organize their files and directories. You can list, move, rename, delete, and create files and directories."
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