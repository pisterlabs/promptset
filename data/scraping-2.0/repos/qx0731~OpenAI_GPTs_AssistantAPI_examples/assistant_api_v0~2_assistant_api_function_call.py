# print out a lot of things to make thing clear 
import os
import time
from datetime import datetime, timedelta
import random
import json
from faker import Faker
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from functions import get_flight_info
_ = load_dotenv(find_dotenv()) # read local .env file
api_key = os.environ['OPENAI_API_KEY']


client = OpenAI(
    api_key =  api_key
)

# step1.A: upload file 
file1 = client.files.create(
    file = open("coupons.tsv", "rb"), 
    purpose = 'assistants'
)
# upload this one for the chatgpt assistant to call the functions 
file2 = client.files.create(
    file = open("functions.py", "rb"), 
    purpose = 'assistants'
)

print(file2.id)

file_list = client.files.list()
print(file_list)

# step1.B: define function 
get_flight_info_json = {
        "name": "get_flight_info",
        "description": "Get flight information between two locations",
        "parameters": {
            "type": "object",
            "properties": {
                "loc_origin": {
                    "type": "string",
                    "description": "The departure airport, e.g. DUS",
                },
                "loc_destination": {
                    "type": "string",
                    "description": "The destination airport, e.g. HAM",
                },
            },
            "required": ["loc_origin", "loc_destination"],
        },
    }

book_flight_json =  {
        "name": "book_flight",
        "description": "Book a flight based on flight information",
        "parameters": {
            "type": "object",
            "properties": {
                "loc_origin": {
                    "type": "string",
                    "description": "The departure airport, e.g. DUS",
                },
                "loc_destination": {
                    "type": "string",
                    "description": "The destination airport, e.g. HAM",
                },
                "datetime": {
                    "type": "string",
                    "description": "The date and time of the flight, e.g. 2023-01-01 01:01",
                },
                "airline": {
                    "type": "string",
                    "description": "The service airline, e.g. Lufthansa",
                },
            },
            "required": ["loc_origin", "loc_destination", "datetime", "airline"],
        },
    }

file_complaint_json = {
        "name": "file_complaint",
        "description": "File a complaint as a customer",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the user, e.g. John Doe",
                },
                "email": {
                    "type": "string",
                    "description": "The email address of the user, e.g. john@doe.com",
                },
                "text": {
                    "type": "string",
                    "description": "Description of issue",
                },
            },
            "required": ["name", "email", "text"],
        },
    }

# def get_flight_info(loc_origin, loc_destination):
#     """Get flight information between two locations."""

#     # Example output returned from an API or database
#     flight_info = {
#         "loc_origin": loc_origin,
#         "loc_destination": loc_destination,
#         "datetime": str(datetime.now() + timedelta(hours=2)),
#         "airline": "KLM",
#         "flight": "KL643",
#     }

#     return json.dumps(flight_info)

# step2: create the assistants with the file 
assistant_instruction = "This assitant is set up in a way to help find flight information, help book flight and file complaints"
assistant = client.beta.assistants.create(
    name = "Test_functional_call", 
    instructions= assistant_instruction, 
    model = 'gpt-4-1106-preview',  
    tools=[{"type":"retrieval"}, 
           {"type": "code_interpreter"}, 
           {"type":"function", "function":get_flight_info_json}, 
           {"type":"function", "function":book_flight_json}, 
           {"type":"function", "function":file_complaint_json}],
    file_ids = [file1.id, file2.id]
)
print(assistant.id) 

my_assistant = client.beta.assistants.list(
    order = 'desc', 
    limit = "20"
)
print(my_assistant.data)

# step3: create a thread 
thread = client.beta.threads.create(
    
)
print(thread)

# step4: add more message to the thread 
message = client.beta.threads.messages.create(
    thread_id = thread.id, 
    role = 'user', 
    content = 'what is the flight information from seattle to austin'
) 

# step5: run the assitant to get the response 
run = client.beta.threads.runs.create(
    thread_id = thread.id, 
    assistant_id= assistant.id
)
print(run.id)

# step6: retrieve the run status 
print(run.status)
while run.status not in ["completed", "failed", "requires_action"]:
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id, 
        run_id = run.id
    )
    print(run.status)
    time.sleep(10)
    

# step 7: 
if run.status == "requires_action": 
    tools_to_call = run.required_action.submit_tool_outputs.tool_calls
    tool_output_array = []
    for tool in tools_to_call:
        tool_call_id = tool.id
        function_name = tool.function.name  # str
        function_arg = json.loads(tool.function.arguments) 
        if function_name == 'get_flight_info':
            chosen_function = eval(function_name)
            output = chosen_function(function_arg["loc_origin"], function_arg["loc_destination"])
        if function_name == 'book_flight':
            chosen_function = eval(function_name)
            output = chosen_function(function_arg["loc_origin"], function_arg["loc_destination"], function_arg["datetime"], function_arg["airline"])
        if function_name == 'file_complaint':
            chosen_function = eval(function_name)
            output = chosen_function(function_arg["name"], function_arg["email"], function_arg["text"])
        tool_output_array.append({"tool_call_id": tool_call_id, "output": output})
        
print(tool_output_array)  

run = client.beta.threads.runs.submit_tool_outputs(
  thread_id=thread.id,
  run_id=run.id,
  tool_outputs= tool_output_array
)
print(run.status)
while run.status not in ["completed", "failed", "requires_action"]:
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id, 
        run_id = run.id
    )
    print(run.status)
    time.sleep(10)
    
# step7: 
messages = client.beta.threads.messages.list(
    thread_id=thread.id
) 

for m in messages: 
    print(m.role + ": " + m.content[0].text.value) 
    print("============================")
