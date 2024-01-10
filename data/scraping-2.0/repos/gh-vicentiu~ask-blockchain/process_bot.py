# process_bot.py
#filename process_bot.py - keep this comment always
import time
import json
import logging
import openai  # Import the OpenAI library for AI-related operations

# Importing necessary functions from other modules
from ai_make.create_ai import create_assistant
from ai_make.create_thread import create_thread  # To create a new conversation thread
from ai_run.send_mess import add_message_to_thread  # To add a message to a conversation thread
from ai_run.run_ai import run_assistant  # To run the AI assistant within a thread
from functions.db_operations import read_db_chats, write_db_chats, read_db_agents, write_db_agents, read_db_assistants, write_db_assistants  # To handle database operations
from functions.ai_parse_response import ai_parse_response
from functions.return_response import send_message_to_hook


client = openai.Client()


# Main function to process a user message
def process_bot(instruction, thread_main):
    user_id = thread_main['u_bot_0_id']
    dbc = read_db_chats(user_id)
    dbb = read_db_assistants(user_id)
    dba = read_db_agents()
    # Log the incoming user ID and message
    logging.info(f"Processing bot {thread_main['agent']}: {thread_main['u_bot_0_id']} with message: {instruction}")
    thread_full = None
    ids = 'active'

    # Retrieve or create an assistant ID for the bot
    assistant_id = dba[ids].get(thread_main['agent'] + '_assistant_id')
    if not assistant_id:
        logging.info(f"Creating new assistant for {thread_main['agent']}.")
        assistant = create_assistant(thread_main['agent'])
        assistant_id = assistant.id
        dba[ids][thread_main['agent'] + "_assistant_id"] = assistant_id
        write_db_agents(dba)
    logging.info(f"Global Assistant {thread_main['agent']}: {assistant_id}.")


    # Retrieve or create a thread ID for the conversation
    dbc = read_db_chats(user_id)
    thread_id = dbb.get('active', {}).get(thread_main['agent'] + '_thread_id')
    if not thread_id:
        logging.info(f"Creating new thread for {thread_main['agent']}.")
        thread_id = create_thread()
        dbb['active'][thread_main['agent'] + '_thread_id'] = thread_id
        write_db_assistants(user_id, dbb)
    logging.info(f"Global Assistant {thread_main['agent']}: {thread_id}.")
   
   
   
    
    message_u_id = add_message_to_thread(thread_id, instruction, role='user', agent=thread_main['agent'])

    logging.info(f"Message {message_u_id} added to  {assistant_id} - {thread_id} for {thread_main['u_bot_0_id']}.")

    dbc = read_db_chats(user_id)    
    if thread_id not in dbc[thread_main['a_bot_0_id']][thread_main['t_bot_0_id']][thread_main['m_bot_0_id']]:
        dbc[thread_main['a_bot_0_id']][thread_main['t_bot_0_id']][thread_main['m_bot_0_id']][thread_id] = {}
    
    if message_u_id not in dbc[thread_main['a_bot_0_id']][thread_main['t_bot_0_id']][thread_main['m_bot_0_id']][thread_id]:
        dbc[thread_main['a_bot_0_id']][thread_main['t_bot_0_id']][thread_main['m_bot_0_id']][thread_id][message_u_id] = {}
    
    dbc[thread_main['a_bot_0_id']][thread_main['t_bot_0_id']][thread_main['m_bot_0_id']][thread_id][message_u_id]['0'] = {"sent": {"role": thread_main['agent'], "content": instruction, "timestamp": int(time.time())}}
    write_db_chats(user_id, dbc)

    # Run the assistant to process the thread and get a response
    thread_main = {
    'a_bot_1_id': assistant_id, 
    't_bot_1_id': thread_id, 
    'm_bot_1_id': message_u_id, 
    'agent': thread_main['agent'], 
    'u_bot_0_id': thread_main['u_bot_0_id'], 
    'a_bot_0_id': thread_main['a_bot_0_id'], 
    't_bot_0_id': thread_main['t_bot_0_id'], 
    'm_bot_0_id': thread_main['m_bot_0_id']
}
    thread_full = run_assistant(thread_main)
    ai_replay = ai_parse_response(thread_full)
    result = send_message_to_hook(user_id=thread_main['u_bot_0_id'], messaged_back=ai_replay)   
    
    # Return the full conversation threads
    dbc = read_db_chats(user_id)
    dbc[thread_main['a_bot_0_id']][thread_main['t_bot_0_id']][thread_main['m_bot_0_id']][thread_id][message_u_id]['1'] = {"replay": {"role": thread_main['agent'], "content": ai_replay, "timestamp": int(time.time())}}
    write_db_chats(user_id, dbc)

    return ai_replay
