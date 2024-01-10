import os
import json
from openai import OpenAI
from connectors.gcal import *  # Import Google Calendar connector functions
from connectors.mail import fetch_emails_from_today


# Load the JSON data from the file
with open('config.json', 'r') as file:
    config = json.load(file)

emails_from_today = fetch_emails_from_today(config["MAIL_ADDRESS"], config["MAIL_PASSWORD"], config["MAIL_SERVER"])

# Constants
GPT_MODEL = "gpt-3.5-turbo"  # GPT model version
SYSTEM_PROMPT = """
    As an AI assistant, you will utilize 'create_gcal_event' function in your tool call to create 
    events in user's calendar for events and appointments that you found in emails that user sent it
    to you. Your task is to analyze the provided emails and identify potential calendar events. When
    you identify an event, use the 'create_gcal_event' function in your tool call functions to schedule
    it in Google Calendar with 'create_gcal_event' function immediately without asking a confirmation from the user. Avoid making assumptions
    or guesses about the content; focus only on the information clearly presented in the emails.

    Note: Do not send any reply to user in all prompts. Just use tools whenever there is at least one potential event in the mail chains.
    """
PLANNING_PROMPT = """
    Mail chain:
    {}
    Note: Do not send any reply to the user after this prompt. Just use tools whenever there is at least one potential event in the mail chain.
""".format(emails_from_today)

# Message chain for the OpenAI chat completion
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": PLANNING_PROMPT}
]

# Tool definition for calendar event creation
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_gcal_event",
            "description": "Create a new event in Google Calendar",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The title of the event"
                    },
                    "location": {
                        "type": "string",
                        "description": "The location of the event"
                    },
                    "description": {
                        "type": "string",
                        "description": "A description of the event"
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start time of the event in ISO format"
                    },
                    "end_time": {
                        "type": "string",
                        "description": "End time of the event in ISO format"
                    }
                },
                "required": ["summary", "start_time", "end_time"]
            },
        },
    }
]

print(f'USER > {PLANNING_PROMPT}')

# Initialize OpenAI client with the API key
client = OpenAI(api_key=config['OPENAI_KEY'])

# Create a chat completion with OpenAI
response = client.chat.completions.create(
    model=GPT_MODEL,
    messages=messages,
    tools=tools,
    tool_choice="auto"  # auto is default, but we'll be explicit
)

# Extract and print the response message content
response_message = response.choices[0].message

print(f'Admina > {response_message.tool_calls}')

# Process tool calls in the response message
tool_calls = response_message.tool_calls
if tool_calls:
    # Mapping of available functions
    available_functions = {
        "create_gcal_event": create_gcal_event
    } 

    # Append the response message to the message chain
    messages.append(response_message)  

    # Execute function calls and append their responses
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(function_args)
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )

    # Create a second chat completion with the updated messages
    second_response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages
    ) 
    print(second_response.choices[0].message.content)
else:
    print("LOG > There wasn't any tool calls.")