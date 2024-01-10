import numpy as np
import pandas as pd
import json
from openai import OpenAI
import streamlit as st

def show_json(obj):
    display(json.loads(obj.model_dump_json()))

responses = ['false']

function_json = {
  "name": "get_specifications",
  "parameters": {
    "type": "object",
    "properties": {
      "specifications": {
        "type": "string",
        "description": "Specifications of the merchandise like name/type, size, color, material etc."
      },
      "quantity": {
        "type": "number",
        "description": "Total required quantity of the merchandise. Needs to be greater than 0."
      },
      "price": {
        "type": "number",
        "description": "Price per unit of the required merchandise. Needs to be greater than 0"
      },
      "num_days": {
        "type": "number",
        "description": "Number of days to fulfill the order.Needs to be greater than 0"
      },
      "need_logo": {
        "type": "string",
        "description": "Description of the logo required on the merchandise. This should include position of the logo, size of the logo, printing method of logo ,color of the logo, etc. Make it NA if no logo is required."
      }
    },
    "required": [
      "specifications",
      "quantity",
      "price",
      "num_days",
      "need_logo"
    ]
  },
  "description": "Extract all the specifications of the merchandise from the user"
}

## initiate client, assistant and thread
client = OpenAI(api_key="")
for i in [i.id for i in client.beta.assistants.list().data if i.name == "Summarization_Assistant_ani"]:
    client.beta.assistants.delete(i)
assistant = client.beta.assistants.create(
    name="Summarization_Assistant_ani",
    instructions="You are an AI assistant that is taking in procurement request from the users. There are usually for merchandise like hoodies, shirts, mugs and bottles.Your job is to get all the specifications of the merchandize from the user",
    model="gpt-4-1106-preview",
    tools=[
        {"type": "function", "function": function_json},
    ],
)

MATH_ASSISTANT_ID = assistant.id  
thread = client.beta.threads.create()

## necessary functions
def submit_message(client,assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )


def get_response(client,thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")


import time

# # Pretty printing helper
# def pretty_print(messages):
#     print("# Messages")
#     for m in messages:
#         print(f"{m.role}: {m.content[0].text.value}")
#     print()
def pretty_print(messages):
    result = ""
    for m in messages.data[-1:]:
        result += f"{m.content[0].text.value}\n"
    result += "\n"
    return result

# Waiting in a loop
def wait_on_run(client,run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def check_response(client,thread,run):
        # Extract single tool call
    tool_call = run.required_action.submit_tool_outputs.tool_calls[0]
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    responses = ['true'] if arguments['quantity'] > 0 and arguments['price'] > 0 and arguments['num_days'] > 0 else ['false']

    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=[
            {
                "tool_call_id": tool_call.id,
                "output": json.dumps(responses),
            }
        ],
    )
    
    run = wait_on_run(client,run, thread)
    print(run.status)
    completion = True if responses[0] == 'true' else False
    return pretty_print(get_response(client,thread)), completion

