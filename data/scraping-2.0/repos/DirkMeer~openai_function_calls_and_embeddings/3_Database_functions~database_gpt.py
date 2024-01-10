import json
import sqlite3
from pathlib import Path

from decouple import config
from openai import OpenAI

from database_util import Database
from function_description import describe_get_info_from_database
from printer import ColorPrinter as Printer
from prompt_setup import database_query_bot_setup


MODEL = "gpt-3.5-turbo-1106"
client = OpenAI(api_key=config("OPENAI_API_KEY"))
current_directory = Path(__file__).parent
conn = sqlite3.connect(current_directory / "database.sqlite")

company_db = Database(conn)
database_schema: str = str(company_db.get_database_info())
print(database_schema)


def get_info_from_database(query) -> str:
    try:
        res = company_db.execute(query)
        return json.dumps(res)
    except Exception as e:
        return f"Error executing query: {e}, please try again, passing in a valid SQL query in string format as only argument."


def ask_company_db(query):
    messages = [
        {"role": "system", "content": database_query_bot_setup},
        {"role": "user", "content": query},
    ]
    tools = [describe_get_info_from_database(database_schema)]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "get_info_from_database"},
        },
    )

    response_message = response.choices[0].message
    messages.append(response_message)

    while response_message.tool_calls:
        tool_calls = response_message.tool_calls
        available_functions = {
            "get_info_from_database": get_info_from_database,
        }
        for call in tool_calls:
            func_name: str = call.function.name
            func_to_call = available_functions[func_name]
            func_args: dict = json.loads(call.function.arguments)
            func_response = func_to_call(**func_args)

            messages.append(
                {
                    "tool_call_id": call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": func_response,
                }
            )

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        messages.append(response_message)

    Printer.color_print(messages)
    return response_message.content


print(
    ask_company_db(
        "What is the name of the user who got the greatest number of 'helpful' votes over all their reviews combined?"
    )
)
