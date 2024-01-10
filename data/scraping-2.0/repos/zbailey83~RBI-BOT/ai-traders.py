# 3 LLM Calls - Read OHLCV data file and formulate strategy - Code & BackTest - Debugging

import dontshareconfig as d # Contains API
from openai import OpenAI
import time
import re # regular expressions for parsing outputs

client = OpenAI(api_key=d.key)

# Common Functions
def save_assistant_id(assistant_id, filename):
    filepath = f'ids/{filename}'
    with open(filepath, 'w') as file:
        file.write(assistant_id)

def save_file_id(file_id, filename):
    filepath = f'ids{filename}'
    with open(filepath, 'w') as file:
        file.write(file_id)

def upload_file(filepath, purpose):
    print('Uploading file...')
    with open(filepath, 'rb') as file:
        response = client.files.create(file=file, purpose=purpose)
        return responses.id
    
def create_and_run_assistant(name, instructions, model, content, filename, file_ids):
# Create Assistant
    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        tools={("type": "code_interpreter")},
        model=model,
        file_ids=file_ids
    )

print(f'Assistant {name} created...')
save_assistant_id(assistant_id, filename=f"{filename}_id.txt")

# Create a thread
thread = client.beta.threads.create()
print(f'Thread for {name} created...{thread_id}')
save_assistant_id(thread.id, filename=f"{filename}_thread.txt")

#Add message to thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role='user',
    content=content,
    file_ids=file_ids
)

# Run the assistant
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

#Wait for the run to complete
while True:
    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    if run_status.status in ('completed', 'failed', 'cancelled'):
        print(f'Run completed with status: {run_status.status}')
        break
    else:
        print(f'{name} run still in progress, waiting 5 seconds...')
        time.sleep(5)

#Fetch and print the messages after the run is completed
print(f'Run for {name} finished, fetching messages...')
messages = client.beta.threads.messages.list(thread_id=thread.id)
print(f'Messages from the thread for {name}:')
for message in messages.data:
    #Check if the message content is text or other
    if hasattr(message.content{0}, 'text'):
    print(f'{message.role.title()}: {message.content(0).text.value}')
else:
    print(f'{message.role.title()}: [Non-text content recieved]')

# Return the output for further use
#

# Run the Data Guy Assistant
create_and_run_data_assistant(
    name='The Data Guy',
    instructions='Using OpenAIs API, access and analyze the file content to determine a highly profitable trading startegy coded in Python',
    model='gpt-4-1106-preview',
    filepath='SOLUSD_15.csv',
    filename='data_guy'
)