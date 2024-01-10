# process_user.py
#filename process_user.py - keep this comment always
import time
import json
import logging
import openai  # Import the OpenAI library for AI-related operations

# Importing necessary functions from other modules
from ai_make.create_ai import create_assistant  # To create a new AI assistant
from ai_make.create_thread import create_thread  # To create a new conversation thread
from ai_run.send_mess import add_message_to_thread  # To add a message to a conversation thread
from ai_run.run_ai import run_assistant  # To run the AI assistant within a thread
from functions.db_operations import read_db_chats, write_db_chats, read_db_agents, write_db_agents, read_db_assistants, write_db_assistants  # To handle database operations
from functions.ai_parse_response import ai_parse_response
from functions.return_response import send_message_to_hook


client = openai.Client()

# Main function to process a user message
def process_user(user_id, messaged_us):
    # Log the incoming user ID and message
    logging.info(f"Processing user: {user_id} with message: {messaged_us}")
    # Read the current state of the database
    dba = read_db_agents()
    dbb = read_db_assistants(user_id)
    dbc = read_db_chats(user_id)

    # Initialize variable for the full thread
    thread_full = None
    last_assistant_id = None
    ids = 'active'

   
    if ids not in dba:
        dba[ids] = {}

    # Retrieve or create an assistant ID for the user
    assistant_id = dba[ids].get('relay_assistant_id')
    if not assistant_id:
        logging.info(f"Creating new global assistant.")
        assistant = create_assistant("relay")
        assistant_id = assistant.id
        dba[ids]['relay_assistant_id'] = assistant_id
        write_db_agents(dba)
    logging.info(f"Assistant {assistant_id} for {user_id}.")
    
   
    # Retrieve or create a thread ID for the conversation
    dbb = read_db_assistants(user_id)
    thread_id = dbb.get('active', {}).get('relay_thread_id')
    if not thread_id:
        logging.info(f"Creating new thread for the user {user_id}.")
        thread_id = create_thread()  # Your function to create a new thread ID
        if 'active' not in dbb:
            dbb['active'] = {}
        dbb['active']['relay_thread_id'] = thread_id
        write_db_assistants(user_id, dbb)
        if assistant_id not in dbc:
            dbc[assistant_id] = {}
        dbc[assistant_id][thread_id] = {}
        write_db_chats(user_id, dbc)
        logging.info(f"Thread {thread_id} for {user_id}.")

    
    logging.info(f"Adding Message to  {assistant_id} - {thread_id} for {user_id}.")
    message_u_id = add_message_to_thread(thread_id, messaged_us, role='user', agent=None)
   
    logging.info(f"Message {message_u_id} added to  {assistant_id} - {thread_id} for {user_id}.")
   
    dbc = read_db_chats(user_id)  
    if message_u_id not in dbc[assistant_id][thread_id]:
        dbc[assistant_id][thread_id][message_u_id] = {}
    dbc[assistant_id][thread_id][message_u_id]['0'] = {"sent": {"role": "relay", "content": messaged_us, "timestamp": int(time.time())}}
    write_db_chats(user_id, dbc)
      

    # Run the assistant to process the thread and get a response
    logging.info(f"Start Main Assistent: 'u_bot_0_id': {user_id}, 'a_bot_0_id': {assistant_id}, 't_bot_0_id': {thread_id}, 'm_bot_0_id': {message_u_id}, 'agent': 'relay'")
    thread_main = {'u_bot_0_id': user_id, 'a_bot_0_id': assistant_id, 't_bot_0_id': thread_id, 'm_bot_0_id': message_u_id, 'agent': 'relay'}
    thread_full = run_assistant(thread_main)
    ai_replay = ai_parse_response(thread_full)
    result = send_message_to_hook(user_id, messaged_back=ai_replay)
    
    dbc = read_db_chats(user_id)
    # Return the full conversation threads
    dbc[assistant_id][thread_id][message_u_id]['1'] = {"replay": {"role": thread_main['agent'], "content": ai_replay, "timestamp": int(time.time())}}
    write_db_chats(user_id, dbc)
    return ai_replay
