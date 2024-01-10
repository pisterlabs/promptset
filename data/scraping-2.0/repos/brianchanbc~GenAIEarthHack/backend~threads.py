import re
from openai import OpenAI
import dotenv
import os
import streamlit as st
import time
from .utils import extract_json

dotenv.load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

client = OpenAI()


def create_assistant_file(uploaded_file):
    assistant_file = client.files.create(file=uploaded_file, purpose="assistants")
    return assistant_file


def _create_assistant(name, instructions, assistant_files=[]):
    if len(assistant_files) != 0:
        assistant = client.beta.assistants.create(
            instructions=instructions,
            name=name,
            tools=[{"type": "retrieval"}],
            model=OPENAI_MODEL,  # type: ignore
            file_ids=[file.id for file in assistant_files],
        )
    else:
        assistant = client.beta.assistants.create(
            instructions=instructions,
            name=name,
            model=OPENAI_MODEL,  # type: ignore
        )
    return assistant


def _update_assistant(
    assistant_id,
    instructions,
    assistant_files=[],
):
    if len(assistant_files) != 0:
        return client.beta.assistants.update(
            assistant_id=assistant_id,
            instructions=instructions,
            tools=[{"type": "retrieval"}],
            file_ids=[file.id for file in assistant_files],
        )

    return client.beta.assistants.update(
        assistant_id=assistant_id, instructions=instructions, tools=[]
    )


def _get_assistant_id(assistant_name):
    all_assistants = client.beta.assistants.list(limit="10")  # type: ignore
    for assistant in all_assistants.data:
        if assistant.name == assistant_name:
            return assistant.id
    return None


def _retrieve_assistant(assistant_id):
    assistant = client.beta.assistants.retrieve(assistant_id)
    return assistant


def _create_thread():
    thread = client.beta.threads.create()
    return thread


def _create_message(message, thread):
    return client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message,
    )


def retrieve_latest_message_content(thread):
    message = client.beta.threads.messages.list(thread_id=thread.id, limit=1)
    message_content = message.data[0].content[0].text.value  # type: ignore
    return message_content


def _create_run(thread, assistant):
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )


def _retrieve_run(thread, run):
    return client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)


def process_uploaded_files(uploaded_files) -> list:
    assistant_files = [create_assistant_file(file) for file in uploaded_files]
    return assistant_files


def get_assistant(assistant_name, instructions, problem, solution, uploaded_files):
    """If the assistant exists, retrive the assistant and update it with new attached files.
    If not, create a new assistant."""

    assistant_id = _get_assistant_id(assistant_name)
    instructions = instructions.format(problem_text=problem, solution_text=solution)
    # If assistant exists, update the assistant.
    if assistant_id:
        return _update_assistant(assistant_id, instructions, uploaded_files)

    # If assistant doesn't exists, create a new one.
    return _create_assistant(assistant_name, instructions, uploaded_files)


def generate_response(assistant, prompt, thread=None):
    if not thread:
        thread = _create_thread()

    prompt = _create_message(prompt, thread)
    run = _create_run(thread, assistant)

    while run.status != "completed":
        run = _retrieve_run(thread, run)

    response = retrieve_latest_message_content(thread)

    return thread, response
