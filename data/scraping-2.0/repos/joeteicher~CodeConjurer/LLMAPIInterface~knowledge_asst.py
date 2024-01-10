import openai
from openai import OpenAI
import logging
from LLMAPIInterface.llm_api import *
import json
import os
import time



# Retrieve a logger for the current module
logger = logging.getLogger(__name__)

assistant = None
client = None

def init(project_diretory, records):
    global assistant
    global thread
    global client
    client = get_client()
    asst_prompt = "You are a Knowledge Assistant specialized in software development.\n "
    asst_prompt += "Your role is to manage and organize a comprehensive knowledge base, "
    asst_prompt += "including code snippets, documentation, and contextual information about "
    asst_prompt += "various software tools and components.\n You are adept at retrieving relevant "
    asst_prompt += "information, understanding complex software systems, and providing clear, "
    asst_prompt += "concise explanations.\n Your expertise includes a wide range of programming "
    asst_prompt += "languages, software development methodologies, and current best practices "
    asst_prompt += "in the industry.\n When asked, synthesize information from the knowledge base "
    asst_prompt += "to answer questions, provide code examples, explain concepts, or offer guidance "
    asst_prompt += "on software development tasks. Always ensure your responses are accurate, "
    asst_prompt += "up-to-date, and aligned with the current context of the inquiry.\n"

    assistant = client.beta.assistants.create(
        instructions=asst_prompt,
        model="gpt-4-1106-preview",
        tools=[{"type":"retrieval"}]
    )
    thread = client.beta.threads.create()
    print(assistant.id)
    print(thread.id)
    records["assistant_id"] = assistant.id


def load_assistant(records):
    global assistant
    global thread
    global client
    client = get_client()
    assistant = client.beta.assistants.retrieve(records["assistant_id"])
    thread = client.beta.threads.create()

def init_load_files(project_directory):
    records = read_record(f"{project_directory}/knowledge/record.json")
    if records["assistant_id"] == None:
        init(project_directory)
    else:
        load_assistant(records)


def read_record(record_file):
    """
    Reads the record file to get the assistant's ID and the list of loaded files.
    Creates the record file with default data if it doesn't exist.

    Args:
        record_file: str, path to the record file.

    Returns:
        dict: A dictionary containing the assistant's ID and the list of loaded files.
    """
    if not os.path.exists(record_file):
        with open(record_file, 'w') as file:
            default_data = {"assistant_id": None, "loaded_files": []}
            json.dump(default_data, file)
            return default_data
    else:
        with open(record_file, 'r') as file:
            return json.load(file)


def update_record(record_file, assistant_id, loaded_files):
    """
    Updates the record file with the assistant's ID and the list of loaded files.

    Args:
        record_file: str, path to the record file.
        assistant_id: str, the ID of the assistant.
        loaded_files: list, a list of loaded file names.
    """
    with open(record_file, 'w') as file:
        json.dump({"assistant_id": assistant_id, "loaded_files": loaded_files}, file)

def load_new_files(project_directory, record_file):
    """
    Loads new files from the project directory into the assistant.

    Args:
        project_directory: str, path to the project directory.
        record_file: str, path to the record file.
    """
    record = read_record(record_file)
    assistant_id = record["assistant_id"]
    loaded_files = record["loaded_files"]

    for file_name in os.listdir(project_directory):
        file_path = os.path.join(project_directory, file_name)
        if os.path.isfile(file_path) and file_name not in loaded_files:
            # Load the file into the assistant (assuming a function load_file exists)
            load_file(assistant_id, file_path)
            loaded_files.append(file_name)

    for file_name in os.listdir(f"{project_directory}/knowledge"):
        file_path = os.path.join(project_directory, file_name)
        if not os.path.isfile(file_path):
            # Load the file into the assistant (assuming a function load_file exists)
            load_file(assistant_id, file_path)
            loaded_files.append(file_name)

    # Update the record with the new list of loaded files
    update_record(record_file, assistant_id, loaded_files)

def load_file(file_path):
    """
    Loads a file into the assistant.

    Args:
        assistant_id: str, the ID of the assistant.
        file_path: str, path to the file.
    """
    global assistant
    # Upload a file with an "assistants" purpose
    file = client.files.create(
        file=open(file_path, "rb"),
        purpose='assistants'
    )
    # Load the file into the assistant
    assistant.file_ids.append(file.id)
    
def analyze_code_dependencies(file_name):
    """
    Analyzes a Python file for missing dependencies and inconsistencies, considering the entire project context.

    Args:
        file_name: str, the name of the file to be analyzed.
        project_files: dict, a dictionary where keys are file names and values are the content of these files.
        system_context: str, a description of the system or project context.

    Returns:
        str: Analysis result highlighting missing dependencies, incorrect function calls, and other inconsistencies.
    """
    with open(file_name, 'r') as f:
        file_content = f.read()

    system_prompt = "Analyze the following Python file for any missing dependencies, incorrect function calls, "
    system_prompt += "or other inconsistencies. Consider the context of related project files and the overall system. "
    system_prompt += "Provide a detailed analysis that includes what is missing or inconsistent, "
    system_prompt += "and suggest corrections where applicable.\n\n"
    system_prompt += "Python File Content:\n" + file_content + "\n\n"

    response = send_text_request(system_prompt)
    return parse_text_response(response)

def send_text_request(system_prompt):
    global client
    global assistant
    thread = client.beta.threads.create()
    message = client.beta.messages.create(
        thread_id = thread.id,
        role="user",
        assistant_id=assistant.id
    )
    run = client.beta.completions.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    time.sleep(2)  # Sleep for 2 seconds
    message = client.beta.messages.retrieve(
        thread_id=thread.id,
        message_id=message.id
    )
    while message.status != "completed":
        time.sleep(2)  # Sleep for 2 seconds
        message = client.beta.messages.retrieve(
            thread_id=thread.id,
            message_id=message.id
        )
    return message

def parse_text_response(response):
    """
    Parses the response received from the LLM API.
    :param response: dict, the response from the API.
    :return: str, the text generated by the LLM.
    """
    #print(response.choices[0].message.content)
    logging.info(response.choices[0].message.content)
    return response.choices[0].message.content