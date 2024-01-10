import os
import json
import time
from pyexpat.errors import messages
from flask import Flask, redirect, render_template, request, jsonify, session, url_for
from openai import OpenAI
import datetime

# Load configuration from config.json
with open('config.json') as f:
    config = json.load(f)

ASSISTANT_NAME = config["ASSISTANT_NAME"]
ASSISTANT_ROLE = "\n".join(config["ASSISTANT_ROLE"])

# Initialize OpenAI API
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def list_assistants():

    assistant_object = client.beta.assistants.list()
    return assistant_object

#def delete_assistant(assistant_id):
#    """Delete an assistant by ID."""
#    delete_url = f"{BASE_URL}/{assistant_id}"
#    response = requests.delete(delete_url, headers=HEADERS)
#    if response.status_code == 200:
#        print(f"Deleted assistant with ID: {assistant_id}")
#    else:
#        print(f"Failed to delete assistant with ID: {assistant_id}. Status Code: {response.status_code}")
#
#def delete_all_assistants():
#    """Delete all assistants."""
#    a_list = list_assistants()
#    assitant_obj_list = a_list.data
#    for i in range(len(assitant_obj_list)):
#        delete_assistant(assitant_obj_list[i].id)

def select_assistant(assistant_id):
    # Use the 'beta.assistants' attribute, not 'Assistant'
    assistant = client.beta.assistants.retrieve(assistant_id)
    return assistant.id

def create_assistant(name, instructions, tools, model):
    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        tools=tools,
        model=model
    )
    return assistant.id  # Return the assistant ID



def get_assistant_by_id(assistant_id):
    assistant = client.beta.assistants.retrieve(assistant_id)
    return assistant.id


def create_thread():
    
    thread = client.beta.threads.create()
    return thread


def select_assistant(assistant_id):
    return get_assistant_by_id(assistant_id)

print("List of assistants:")
assistants = list_assistants()

for i in range(len(assistants.data)):
    ass = assistants.data[i]
    print(i)
    print(ass.name)
    print(ass.id)
    print(ass.instructions)