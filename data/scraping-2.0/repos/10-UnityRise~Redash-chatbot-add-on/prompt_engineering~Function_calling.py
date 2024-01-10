import openai
import os
import json
import requests
from pprint import pprint
import json
import openai
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import re

GPT_MODEL = "gpt-3.5-turbo-0613"

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
_ = load_dotenv(find_dotenv())
openai.api_key  = os.getenv('OPENAI_API_KEY')


import sys 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect 

username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
database = os.getenv("DB_DATABASE")

database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
engine = create_engine(database_url)
Session = sessionmaker(bind=engine)
conn = engine.connect()

print("Opened database successfully")

print("Opened database successfully")



@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )

        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


# In[5]:


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tool": "magenta",
    }

    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "tool":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))


# In[6]:


def get_table_names(engine):
    """Return a list of table names."""
    table_names = []
    inspector = inspect(engine)
    for table_name in inspector.get_table_names():
        table_names.append(f'"{table_name}"')  # Add double quotes around table name
    return table_names

def get_column_names(engine, table_name):
    """Return a list of column names."""
    column_names = []
    inspector = inspect(engine)
    for column in inspector.get_columns(table_name):
        column_names.append(f'"{column["name"]}"')  # Add double quotes around column name
    return column_names

def get_database_info(engine):
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    inspector = inspect(engine)
    for table_name in inspector.get_table_names():
        columns_names = get_column_names(engine, table_name)
        table_dicts.append({"table_name": f'"{table_name}"', "column_names": columns_names})  # Add double quotes around table name
    return table_dicts


# In[7]:


database_schema_dict = get_database_info(conn)
database_schema_string = "\n".join(
    [
        f'Table: "{table["table_name"]}"\nColumns: {", ".join([f"{col}" for col in table["column_names"]])}'
        for table in database_schema_dict
    ]
)


# In[8]:


tools = [
    {
        "type": "function",
        "function": {
            "name": "ask_database",
            "description": "Use this function to answer user questions about youtube. Input should be a fully formed SQL query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"""
                                SQL query extracting info to answer the user's question.
                                SQL should be written using this database schema:
                                {database_schema_string}
                                The query should be returned in plain text, not in JSON.
                                """,
                    }
                },
                "required": ["query"],
            },
        }
    }
]


# In[9]:


def ask_database(engine, query):
    """Function to query PostgreSQL database with a provided SQL query."""
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            results = result.fetchall()
    except Exception as e:
        results = f"Query failed with error: {e}"
    return results
    
def execute_function_call(message, engine):
    if message["tool_calls"][0]["function"]["name"] == "ask_database":
        query = json.loads(message["tool_calls"][0]["function"]["arguments"])["query"]
        results = ask_database(engine, query)
    else:
        results = f"Error: function {message['tool_calls'][0]['function']['name']} does not exist"
    return results


# In[12]:


messages = []
messages.append({"role": "system", "content": "Answer user questions by generating SQL queries against the youtube data Database."})
messages.append({"role": "user", "content": "Hi, who are the top 5 cities by number of viewers?"})
chat_response = chat_completion_request(messages, tools)
print("===================",chat_response.json())
assistant_message = chat_response.json()["choices"][0]["message"]
assistant_message['content'] = str(assistant_message["tool_calls"][0]["function"])
print("===================",assistant_message['content'])
messages.append(assistant_message)
if assistant_message.get("tool_calls"):
    results = execute_function_call(assistant_message, engine)
    messages.append({"role": "tool", "tool_call_id": assistant_message["tool_calls"][0]['id'], "name": assistant_message["tool_calls"][0]["function"]["name"], "content": results})
pretty_print_conversation(messages)


# In[11]:


messages.append({"role": "user", "content": "What is the name of the city with the most views?"})
chat_response = chat_completion_request(messages, tools)
assistant_message = chat_response.json()["choices"][0]["message"]
assistant_message['content'] = str(assistant_message["tool_calls"][0]["function"])
messages.append(assistant_message)
if assistant_message.get("tool_calls"):
    results = execute_function_call(assistant_message, engine)
    messages.append({"role": "tool", "tool_call_id": assistant_message["tool_calls"][0]['id'], "name": assistant_message["tool_calls"][0]["function"]["name"], "content": results})
pretty_print_conversation(messages)





