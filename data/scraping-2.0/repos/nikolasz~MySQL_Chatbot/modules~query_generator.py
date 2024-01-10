# query_generator.py
from openai_api import generate_response
import json
import logging

def prepare_system_message(schema, sql_commands):
    return f'''
    You are an SQL query generator. Generate an SQL query based on the following schema, SQL commands, and user's input.
    Schema: {json.dumps(schema, indent=4)}
    SQL Commands: {json.dumps(sql_commands, indent=4)}
    '''

def generate_sql_query(schema, query, sql_commands):
    """Generates an SQL query from a natural language query.

    Args:
        schema (dict): The schema of the database.
        query (str): The natural language query.
        sql_commands (dict): The SQL commands.

    Returns:
        str: The generated SQL query, or 'error' if there was an issue.
    """
    # Prepare the system message
    system_message = prepare_system_message(schema, sql_commands)

    # Prepare the messages
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    # Print system/user messages for debugging
    print(f'System Message: {system_message}')
    print(f'User Message: {query}')

    # Call OpenAI API
    response = generate_response(messages)

    # Error checking
    if not response:
        logging.warning(f"No response received for query: {query}")
        return 'error'
    if 'error' in response:
        logging.error(f"An error occurred while generating the query: {response}")
        return 'error'

    # Return the generated SQL query
    return response
