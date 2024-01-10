from .utils import openai_functions
from .models import ToDoResponse
from pydantic import parse_obj_as
import pandas as pd

"""
This module contains functions for processing emails and writing them to file.
"""

# Extend this to
# Take in a csv of emails including their statuses
# If not yet processed, process them
# Write the output back into the csv
# So that Streamlit frontend ONLY reads from the csv (or DB later on)

def get_to_do_list() -> dict:
    todo_progress_source = "data/todo_progress.csv"
    todo_progress_df = pd.read_csv(todo_progress_source)
    todo_progress_df = todo_progress_df[todo_progress_df['status'] != 'done']

    for index, row in todo_progress_df.iterrows():
        gpt_output = get_gpt_response(row['email_message'])
        print(gpt_output)
        todo_progress_df.loc[index, 'chatgpt_output'] = str(gpt_output)
        if 'next_step' in gpt_output:
            todo_progress_df.loc[index, 'status'] = 'done'
        else:
            todo_progress_df.loc[index, 'status'] = 'error'
    
    todo_progress_df.to_csv(todo_progress_source, index=False)

def get_gpt_response(email_string) -> dict:
    """
    Gets GPT response for a given email string
    """
    system_prompt = _create_system_prompt()
    user_prompt = "Email:\n" + email_string
    response = openai_functions.call_gpt(system_prompt, user_prompt)
    return _check_response(response)

def _create_system_prompt() -> str:
    """
    Returns a prompt asking GPT to parse the email and assess follow-up items
    """
    prompt = f'''You are a helpful personal assistant. Process the email given and return the output in strict JSON with the following fields:
   
    {{
    "next_step": "if a follow-up is absolutely required, return 'todo';
                  if a follow-up is suggested but not absolutely required, return 'review';
                  if no follow-ups are needed, return 'digest'",
    "action": "if next_step is 'todo' or 'review', return the suggested action. Otherwise, return 'None'",
    "deadline": "if next_step is 'todo' or 'review', return the suggested deadline for the action. Otherwise, return 'None'",
    "summary": "summarize the email in two sentences",
    "effort": "estimate the effort level for the action - 'low', 'medium', or 'high'",
    "constraint: "if next_step is 'todo' or 'review', extract any other information that can inform the suggested action, such as office hours, phone number, etc. Return a dictionary wiht the relevant keys, or 'None'" 
    }}
    ''' 
    return prompt

def _check_response(response) -> str:
    """
    Check the GPT response, parse it if necessary
    """
    try:
        response = parse_obj_as(ToDoResponse, response)
        return response.dict()
    except Exception as e:
        return {"Error": str(e)}