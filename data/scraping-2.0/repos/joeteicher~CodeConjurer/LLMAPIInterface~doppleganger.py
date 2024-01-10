from openai import OpenAI
import logging
from LLMAPIInterface.llm_api import *
from LLMAPIInterface.knowledge_asst import read_record, update_record, parse_text_response
import json
import os
import time
import datetime

logger = logging.getLogger(__name__)

class Doppleganger:
    
    def __init__(self):
        self.client = None
        self.assistant = None
        self.thread = None
        self.records = None

def init_doppelganger(client, records):
    global assistant
    instruct_txt = "You are an AI assistant designed to emulate the thought processes,"
    instruct_txt += "writing style, and decision-making approach of a person, based on his personal writings." 
    instruct_txt += "You should use this knowledge to mimic his's style and thought patterns" 
    instruct_txt += "in your responses. When interacting, always be aware that you are communicating"
    instruct_txt += "with the real person. Use this context to tailor your responses appropriately,"
    instruct_txt += "ensuring they reflect what he might say or think in similar situations."
    files = []
    for file_name in os.listdir("C:/Doppleganger/Files"):
        file = client.files.create(file=open(f"C:/Doppleganger/Files/{file_name}", 'rb'), 
                            purpose='assistants')
        files.append(file)
    fids = []
    for file in files:
        fids.append(file.id)
    assistant = client.beta.assistants.create(
        instructions=instruct_txt,
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids = fids
    )
    records["assistant_id"] = assistant.id
    records["loaded_files"] = [fids]
    update_record("C:/Doppleganger/records.json", records["assistant_id"], records["loaded_files"])
    return assistant, records

def load_doppelganger(client, records):
    assistant = client.beta.assistants.retrieve(records["assistant_id"])
    return assistant

def init_or_load():
    minime = Doppleganger()
    records = read_record("C:/Doppleganger/records.json")
    client = get_client()
    if records["assistant_id"] == None:
        assistant, records = init_doppelganger(client, records)
    else:
        assistant = load_assistant(client, records)
    thread = client.beta.threads.create()
    minime.client = client
    minime.assistant = assistant
    minime.thread = thread
    minime.records = records
    return minime
    #load_new_files(records)

def load_new_files(records, client):
    loaded_files = records["loaded_files"]
    for file_name in os.listdir("C:/Doppleganger/Files"):
        if file_name not in loaded_files:
            file = client.files.create(file=open(f"C:/Doppleganger/Files/{file_name}", 'rb'), 
                                purpose='assistants')
            assistant.file_ids.append(file.id)
            loaded_files.append(file_name)
    update_record("C:/Doppleganger/records.json", records["assistant_id"], loaded_files)

def load_assistant(client, records):
    print(f"loading assistant {records['assistant_id']}")
    assistant = client.beta.assistants.retrieve(records["assistant_id"])
    return assistant


def send_text_request(prompt, client, assistant, thread, records):
    message = client.beta.threads.messages.create(
        thread_id = thread.id, 
        role="user",
        content=prompt
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    time.sleep(2)#sleep for 2 seconds
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    while run.status != "completed":
        time.sleep(2)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    response = ""
    for message in messages.data:
        if message.role == "assistant":
            response += message.content[0].text.value
    return response

def iterate_conversation(input_string, minime):
    client = minime.client
    assistant = minime.assistant
    thread = minime.thread
    records = minime.records
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"C:/Doppleganger/Files/conversation_{date_str}.txt"
    with open(filename, 'a' if os.path.exists(filename) else 'w', encoding='utf-8') as file:
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"DateTime: {current_datetime}, Real Me: {input_string}\n")
        response_string = send_text_request(input_string, client, assistant, thread, records)
        file.write(f"DateTime: {current_datetime}, Doppleganger: {response_string}\n")
    return response_string  