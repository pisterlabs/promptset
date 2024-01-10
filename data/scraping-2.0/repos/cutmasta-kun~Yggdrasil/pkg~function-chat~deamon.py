# deamon.py
import logging
import time
import os
import json
import traceback
from conversations_repository import get_all_conversations, get_conversation_from_repo, add_speech_bubble
from openai_client import OpenAIClient
from copy import deepcopy

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MEMORY_HOST = os.getenv('MEMORY_HOST', 'http://memory:8001')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ORGANIZATION = os.getenv('OPENAI_ORGANIZATION')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')

unused_functions = [
    {
        "name": "log",
        "description": "Logs the summary of the conversation so far.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "The summary of the conversation."
                }
            },
            "required": ["summary"]
        }
    }
]

def log_function(summary="", **kwargs):
    return {"summary": summary}

functions = [
    {
        "name": "extract_data",
        "description": "Extracts key-value pairs of information from a user message.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "description": "A dictionary containing extracted key-value pairs from the user message."
                }
            }
        }
    }
]

def extract_data(**kwargs):
    logging.info(f"Identified keys: {kwargs}")

    return {"data": kwargs}

function_callbacks = {
    # "log": log_function,
    "extract_data": extract_data
}

function_messages = {
    # "log": [],
    "extract_data": [
        {
            "content": "Please extract all available explicit and implicit key-value pairs from the following user input. If file contents are provided in recognizable formats like yaml, json, or similar code blocks, extract the entire content of the file as a single string. Use a combination of the primary subject or topic from the user's message and the file type to generate a dynamic key for the extracted content. For example, if the user mentions OpenAI and provides a yaml file, the key could be openai yaml. Always store file contents as strings, not parsed objects. Please use lowercase English definitions for the keys, but retain the exact same case for the values.",
            "role": "system"
        },
    ]
}

if not OPENAI_API_KEY or not OPENAI_ORGANIZATION or not OPENAI_MODEL:
    logger.error('The OPENAI_API_KEY, OPENAI_ORGANIZATION, and OPENAI_MODEL environment variables must be set.')
    exit(1)

openai_client = OpenAIClient(OPENAI_ORGANIZATION, OPENAI_API_KEY)

def get_latest_user_and_assistant_messages(conversation):
    reversed_conversation = conversation[::-1]
    latest_messages = []

    user_found, assistant_found = False, False

    for bubble in reversed_conversation:
        if bubble.role == "user" and not user_found:
            latest_messages.append(bubble)
            user_found = True
        elif bubble.role == "assistant" and not assistant_found:
            latest_messages.append(bubble)
            assistant_found = True

        if user_found and assistant_found:
            break

    return latest_messages[::-1]

def process_conversation(conversation_uuid):
    logger.info(f'new process: {conversation_uuid}')

    conversation = get_conversation_from_repo(MEMORY_HOST, conversation_uuid)

    if not conversation:
        logger.info('No conversation found')
        return

    last_speech_bubble = conversation[-1]
    role = last_speech_bubble.role

    if role != "user":
        logger.info('Last speech bubble not from user')
        return

    filtered_conversation = [
        bubble for bubble in conversation
        if bubble.role in ["user", "assistant"]
        and bubble.content != 'None'
        and (bubble.function_call == 'None' or bubble.function_call is None)
    ]

    try:

        latest_filtered_conversation = get_latest_user_and_assistant_messages(filtered_conversation)

        # First, we call the functions
        function_bubbles = openai_client.create_chat_completion_with_functions(OPENAI_MODEL, latest_filtered_conversation, functions, function_callbacks, function_messages)

        for function_bubble in function_bubbles:
            function_bubble_copy = deepcopy(function_bubble)
            add_speech_bubble(MEMORY_HOST, conversation_uuid, function_bubble_copy)
            filtered_conversation.append(function_bubble_copy)

        response_bubble = openai_client.create_chat_completion(OPENAI_MODEL, filtered_conversation)

        if response_bubble:
            # Add the assistant's response to the conversation
            add_speech_bubble(MEMORY_HOST, conversation_uuid, response_bubble)

    except Exception as e:
        logging.error(f"Error: {e}\n{traceback.format_exc()}")

def execute():
    logger.info('starting job')
    conversation_uuids = get_all_conversations(MEMORY_HOST)
    if not conversation_uuids:
        logger.info('No conversations found')
        logger.info('finished job')
        return
    logger.info(f'Found {len(conversation_uuids)} conversations')
    for conversation_uuid in conversation_uuids:
        process_conversation(conversation_uuid)
    logger.info('finished job')

while True:
    execute()
    time.sleep(20)
