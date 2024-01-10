import os
from dotenv import load_dotenv
import openai
import json
import functions

load_dotenv()
openai.api_key = os.getenv("OPENAIKEY")
url = "https://api.openai.com/v1/chat/completions"
aimessage = []
function_list = [
        {
            "name": "check_weather",
            "description": "Get the current weather from a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the weather from"
                    }
                },
                "required": ["city"]
            }
        },
        {
            "name": "create_task",
            "description": "Creates a task",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content of the task"
                    }
                },
                "required": ["content"]
            }
        },
        {
            "name": "check_task_status",
            "description": "Checks if a task is done",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": "The name of the task"
                    }
                },
                "required": ["task_name"]
            }
        },
        {
            "name": "close_task_by_name",
            "description": "Closes a task with a given name",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": "The name of the task"
                    }
                },
                "required": ["task_name"]
            }
        },
        {
            "name": "get_undone_tasks",
            "description": "Gets all undone tasks and returns the name of them",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": "A list of all undone tasks"
                    }
                }
            },
            "required": []
        }
    ]

def second_request(function_name, content, previous):
    messages = [
        {
            "role": "user",
            "content": previous,
        },
        {
            "role": "assistant",
            "content": "null",
            "function_call": aimessage[0]["function_call"]
        },
        {
            "role": "function",
            "name": function_name,
            "content": content
        },
    ]


    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=function_list,
        function_call="auto"
    )

    message = response.choices[0].message # type: ignore
    return message["content"]



def request(text):
    if text != "":
        message = [{
            "role": "user",
            "content": text,
        }]
        previous = text
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=message,
            functions=function_list,
            function_call="auto"
        )
        message = response.choices[0].message # type: ignore
        aimessage.append(message)
        if "function_call" in message:
            function_call = message["function_call"]
            arguments = json.loads(function_call["arguments"])

            response = None
            function_name = function_call["name"]
            if function_name == "check_weather":
                print("check_weather")
                response = functions.check_weather(arguments["city"])
                return second_request(function_name, response, previous)
            elif function_name == "create_task":
                print("create_task")
                response = functions.create_task(arguments["content"])
                return second_request(function_name, response, previous)
            elif function_name == "check_task_status":
                print("check_task_status")
                response = functions.check_task_status(arguments["task_name"])
                return second_request(function_name, response, previous)
            elif function_name == "close_task_by_name":
                print("close_task_by_name")
                response = functions.close_task_by_name(arguments["task_name"])
                return second_request(function_name, response, previous)
            elif function_name == "get_undone_tasks":
                print("get_undone_tasks")
                response = functions.get_undone_tasks()
                return second_request(function_name, response, previous)

        else:
            return message["content"]
    else:
        return "Sorry, I did not understand you."
