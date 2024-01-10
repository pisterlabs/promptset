"""
Utility functions to support OpenAI usage, file operations, and JSON handling.
"""

import json
import sys
import os
import time
from typing import Any, Dict
import openai
from dotenv import load_dotenv


load_dotenv()


def load_json(file_path: str) -> Dict[str, Any]:
    """ Loads a JSON file from the specified file path. """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        sys.exit(f"Error reading {file_path}: {e}")


def write_to_file(file_path: str, content: str) -> None:
    """ Writes the specified content to the specified file path. """
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def get_api_client() -> openai.Client:
    """ Returns an OpenAI API client. """
    api_key = os.environ.get("OPENAI_API_KEY")
    return openai.OpenAI(api_key=api_key)


def create_thread(client: openai.Client, content: str, assistant_id: str) -> str:
    """ Creates a thread with the specified content and assistant ID. """
    thread = client.beta.threads.create(messages=[{"role": "user", "content": content}])
    client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
    return thread.id


def clean_json_string(json_str: str) -> Dict[str, Any]:
    """ Cleans up the specified JSON string and returns a JSON object. """
    cleaned_str = json_str.strip('"')
    cleaned_str = cleaned_str.replace("```json\n", "").replace("\n```", "")
    cleaned_str = cleaned_str.replace("\\n", "\n").replace('\\"', '"')
    return json.loads(cleaned_str)


def get_response(client: openai.Client, thread_id: str) -> str:
    """ Gets the response from the specified thread ID. """
    retries = 0
    while retries < 5:
        try:
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            assistant_messages = [
                msg for msg in messages.data if msg.role == "assistant"
            ]
            if (
                assistant_messages
                and assistant_messages[0].content
                and assistant_messages[0].content[0].text
            ):
                response = assistant_messages[0].content[0].text.value
                if response:
                    return response
            else:
                time.sleep(10)
                retries += 1
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(10)
            retries += 1
    return "Sorry, I'm having trouble connecting to the API. Please try again later."
