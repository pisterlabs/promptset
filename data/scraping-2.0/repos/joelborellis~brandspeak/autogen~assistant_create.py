import os
import openai
import time
from halo import Halo
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_model: str = os.environ.get("OPENAI_MODEL")
openai.api_key = os.environ.get("OPENAI_API_KEY")
# create client for OpenAI
client = OpenAI(api_key=openai.api_key)

###     file operations

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()
    
###     API functions

def upload_file(filename):
        # Upload a file with an "assistants" purpose
        file = client.files.create(
            file=open(filename, "rb"),
            purpose='assistants'
            )
        return file


def get_assistant(file_id):
     while True:
            try:
                assistant = client.beta.assistants.create(
                        name="Shadow Planner",
                        instructions="You are a general assistant that answers questions on retrieved files.",
                        tools=[{"type": "retrieval"}],
                        model="gpt-4-1106-preview",
                        file_ids=[file_id],
                    )             

                return assistant
            except Exception as yikes:
                print(f'\n\nError communicating with OpenAI: "{yikes}"')
                exit(5)

# Function to wait for a run to complete
def wait_for_run_completion(thread_id, run_id):
    while True:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        print(f"Current run status: {run.status}")
        if run.status in ['completed', 'failed', 'requires_action']:
            return run
        
# Function to print messages from a thread
def print_messages_from_thread(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    for msg in reversed(messages.data):
        print(f"{thread_id}:  {msg.role}: {msg.content[0].text.value}")

if __name__ == '__main__':
    # Read the system messgae from file
    #system_message = open_file('./backend/prompts/system_insights.md')

    # upload a file
    file = upload_file("./data/Joel_Borellis_RESUME.pdf")

    # get an assistant
    assistant = get_assistant(file.id)

    # create a thread
    thread = client.beta.threads.create()

    while True:
     # Get user query
        query = input('\n\nQUERY: ').strip()
        if query.lower() == 'exit':
            exit(0)

        message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )
        run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
                )
        
        run = wait_for_run_completion(thread.id, run.id)
        
        if run.status == 'failed':
            print(run.error)
            continue
        elif run.status == 'requires_action':
            run = wait_for_run_completion(thread.id, run.id)

        # Print messages from the thread
        print_messages_from_thread(thread.id)

        