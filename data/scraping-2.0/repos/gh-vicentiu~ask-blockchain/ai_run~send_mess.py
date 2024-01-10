import openai
import logging
import json
import time


client = openai.Client()

def add_message_to_thread(thread_id, messaged_us, role='user', agent=None):
    logging.info(f"{agent} Attempting to add message to thread: {messaged_us}")

    # First, check the status of any active run
    get_runs = client.beta.threads.runs.list(thread_id=thread_id, limit=1, order='desc')
    if get_runs.data:
        run_id = get_runs.data[0].id
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run_status.status not in ['completed', 'failed', 'cancelled', 'expired']:
            logging.info(f"Active run detected. Status: {run_status.status}. Attempting to cancel.")
            # Try to cancel the active run
            cancel_job = client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run_id)
            time.sleep(2)  # Wait for a while before retrying

    # Now, try to add the message
    try:
        added_message = client.beta.threads.messages.create(thread_id=thread_id, role=role, content=messaged_us)
        if added_message.id:
            logging.info(f"{agent} Message ID registered: {added_message.id} - {messaged_us}")
            return added_message.id
        else:
            logging.error(f"{agent} Failed to add message to thread.")
            return None
    except openai.BadRequestError as e:
        logging.error(f"{agent} Error in add_message_to_thread: {e}")
        return None
