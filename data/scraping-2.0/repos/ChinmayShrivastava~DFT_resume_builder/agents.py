from openai import OpenAI
import os
# load dotenv in the base root
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def add_files(file_names_raw):
    files = []
    for file_name_raw in file_names_raw:
        file = client.files.create(
            file=open(file_name_raw["file_name"], 'rb'),
            purpose='assistants'
            )
        files.append(file.id)
    return files

def create_assistant(name, description, file_ids, model='gpt-3.5-turbo-1106'):
    assistant = client.beta.assistants.create(
        name=name,
        description=description,
        model=model,
        tools=[{"type": "retrieval"}],
        file_ids=file_ids
    )
    return assistant

def create_thread(messages_raw):
    messages = []
    for message_raw in messages_raw:
        message = {
            "role": message_raw["role"],
            "content": message_raw["content"]
        }
        messages.append(message)
    thread = client.beta.threads.create(
        messages=messages
    )
    return thread

def add_message(threadid, message_raw):
    message = client.beta.threads.messages.create(
        thread_id=threadid,
        role=message_raw["role"],
        content=message_raw["content"],
    )
    return message

def run_assistant(assistantid, threadid):
    run = client.beta.threads.runs.create(
        thread_id=threadid,
        assistant_id=assistantid
    )
    return run

def get_run(runid, threadid):
    run = client.beta.threads.runs.retrieve(
        thread_id=threadid,
        run_id=runid
    )
    return run

def get_response(threadid):
    messages = client.beta.threads.messages.list(
        thread_id=threadid
    )
    return messages