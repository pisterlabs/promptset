from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
import json
from openai import OpenAI
import time
import numpy as np
os.environ['OPENAI_API_KEY'] = ''
function_json = {
  "name": "get_vendor_quotation",
  "parameters": {
    "type": "object",
    "properties": {
      "requirements_satisfied": {
        "type": "boolean",
        "description": "Can the requirements be satisfied by the vendor? Make it true even if some requirements can be satisfied."
      },
      "unfullfiled_requirements": {
        "type": "string",
        "description": "What requirements cannot be satisfied by the vendor? Separate requirements with ';'. NA if there are no missing requirements."
      },
      "quotation": {
        "type": "number",
        "description": "Price per unit of the product. Only set to NA if the requirements cannot be satisfied at all. Otherwise ask the user for the exact price."
      },
      "procurement_days": {
        "type": "number",
        "description": "Number of days to fullfil the requirements. Only set to NA if the requirements cannot be satisfied at all. Otherwise ask the user for the days"
      }
    },
    "required": [
      "requirements_satisfied",
      "unfullfiled_requirements",
      "quotation",
      "procurement_days"
    ]
  },
  "description": "Check if the vendor can fullfil the requirements and get the vendor quotation and procurement days from the mail."
}


specification = json.loads(json.load(open('C:/VS code projects/Road to Hack/auto_negotiator/Utilities/requirements.json','r')))
client = OpenAI(api_key="")
for i in [i.id for i in client.beta.assistants.list().data if i.name == "Summarization_Assistant_ani"]:
    client.beta.assistants.delete(i)
assistant = client.beta.assistants.create(
    name="Summarization_Assistant_ani",
    instructions=f"You are an AI assistant who is supposed to get the quotation and time of delivery from the vendor. These are the requirements:{specification} You are supposed to collect the three main information: Can the requirements be met, what is the price and how long will it take to deliver (in days). All these requirements need to be provided by the vendor in chat. Do not assume zero. Always write your responses in form of a mail on behalf of Vishwa Singh.",
    model="gpt-4-1106-preview",
    tools=[
        {"type": "function", "function": function_json},
    ],
)

MATH_ASSISTANT_ID = assistant.id  
thread = client.beta.threads.create()

def gpt_draft_mail(specification, vendor_name):
  # Create a GPT prompt
  prompt = f"Write a mail to a vendor named '{vendor_name} on behalf of Vishwa Mohan Singh (salutations), asking for a quotation for the following specifications:\nSpecifications: {specification['specifications']}\nQuantity: {specification['quantity']}\nOur Price: {specification['price']} Euros\nNumber of days Required: {specification['num_days']}\nNeed logo: {specification['need_logo']}\n\nMail:"
  mail_assistant = ChatOpenAI()
  messages = [
      SystemMessage(
          content="You are an AI assistant that is supposed to write a mail to the vendor asking for a quotation and time of delivery. Specify our required price and days as well."
      ),
      HumanMessage(content=prompt),
  ]
  response = mail_assistant(messages)
  return response.content

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
    
    responses = ['true'] if (arguments['quotation'] == 'NA' and arguments['procurement_days']== 'NA') or (arguments['quotation'] > 0 and arguments['procurement_days'] > 0) else ['false']


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
    completion = True if responses[0] == 'true' else False
    run = wait_on_run(client,run, thread)
    print(run.status)
    return pretty_print(get_response(client,thread)), completion

