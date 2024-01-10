import time
import openai
import json
import logging
from ai_tools.main_tools import call_agent_webhook, call_agent_coder
from ai_tools.secondary_tools import execute_file, create_file, move_files
from ai_tools.tool_calls import handle_add_to_webhook, handle_remove_webhook, handle_edit_webhook, handle_test_webhook, handle_call_agent_webhook, handle_call_agent_coder, handle_create_file, handle_execute_file, handle_move_files
from functions.db_operations import read_db_chats, write_db_chats  # To handle database operations
from functions.return_response import send_message_to_hook



client = openai.Client()  # Initialize the OpenAI client
# Read the current state of the database



def run_assistant(thread_main):

    if thread_main['agent'] == 'relay':
        thread_id=thread_main['t_bot_0_id'] 
        assistant_id=thread_main['a_bot_0_id']
        message_u_id=thread_main['m_bot_0_id']
        logging.info("Starting the main assistant...")
        
    else:
        thread_id=thread_main['t_bot_1_id'] 
        assistant_id=thread_main['a_bot_1_id']
        message_u_id=thread_main['m_bot_1_id']
        logging.info("Starting the secondery bots...")
    user_id = thread_main['u_bot_0_id']
    agent = thread_main['agent']
    
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id, instructions="")
    


    logging.info("Main Assistant run initiated. Dumping initial run status:")
    #logging.info(json.dumps(run, default=str, indent=4))

    

    while True:
        logging.info("Checking run status...")
        time.sleep(3)
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        #logging.info(F"Current run status: {run_status}")
        #logging.info(json.dumps(run_status, default=str, indent=4))

        if run_status.status == 'completed':
            logging.info("Run completed. Fetching messages...")
            messages = client.beta.threads.messages.list(thread_id=thread_id, limit=1, order='desc')
            logging.info(f"Messages fetched from the thread {messages}.")
            return messages
 
        elif run_status.status == 'requires_action':
            logging.info("Function Calling")
            required_actions = run_status.required_action.submit_tool_outputs.model_dump()
            logging.info(required_actions)
            tool_outputs = []

            for action in required_actions["tool_calls"]:
                func_name = action['function']['name']
                arguments = json.loads(action['function']['arguments'])
                action_id = action['id']  # Extract the action ID
                result = send_message_to_hook(user_id, messaged_back=(f"{thread_main['agent']}, '{func_name}'"))

                # Refactored: Using a dictionary to map function names to handler functions
                handlers = {
                    "call_agent_webhook": handle_call_agent_webhook,
                    "call_agent_coder": handle_call_agent_coder,
                    "create_file": handle_create_file,
                    "execute_file": handle_execute_file,
                    "move_files": handle_move_files,
                    "add_to_webhook": handle_add_to_webhook,
                    "remove_webhook": handle_remove_webhook,
                    "edit_webhook": handle_edit_webhook,
                    "test_webhook": handle_test_webhook
                }

                if func_name in handlers:
                    handlers[func_name](arguments, thread_main, tool_outputs, action_id)
                    result = send_message_to_hook(user_id, messaged_back=(f"'{tool_outputs}'"))
                    if agent == 'relay':
                        dbc = read_db_chats(user_id)
                        dbc[thread_main['a_bot_0_id']][thread_main['t_bot_0_id']][thread_main['m_bot_0_id']]['2'] = {"tool":{func_name: tool_outputs, "timestamp": int(time.time())}}
                        write_db_chats(user_id, dbc)
                    else:
                        dbc = read_db_chats(user_id)
                        dbc[thread_main['a_bot_0_id']][thread_main['t_bot_0_id']][thread_main['m_bot_0_id']][thread_id][message_u_id]['3'] = {"tool":{func_name: tool_outputs, "timestamp": int(time.time())}}
                        write_db_chats(user_id, dbc)
                        

            print("Submitting outputs back to the Assistant...")
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )

            logging.info(f"Submitting outputs back: {tool_outputs}")
            #logging.info(json.dumps(run, default=str, indent=4))


        elif run_status.status == 'failed':
            logging.error("Run failed. Exiting...")
            if run_status.last_error:
                # Directly access the 'message' attribute of last_error
                error_message = run_status.last_error.message if run_status.last_error.message else 'Unknown error'
                logging.error(f"Error details: {error_message}")
            return None

        else:
            logging.info("Waiting for the Assistant to process...")
            time.sleep(3)
    # Update the database with the new state

    return None
