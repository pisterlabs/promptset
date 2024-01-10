import os
import requests
import json
from openai import OpenAI
import datetime
import time

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
global_thread_id = None

# Main function to handle the entire process
def main(user_message, api_key, email):
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Start a new thread and add default messages
    thread = start_thread_and_add_default_msg(client, email)

    # Submit user message
    run, last_message = submit_message(client, thread, user_message)

    run = wait_on_run(client, run, thread)
    # print("run status: ", run.status)
    if run.status == "requires_action":
        print("requires action, submitting outputs")
        submit_outputs(client, thread, run)
        run = wait_on_run_after_requires_action(client, run, thread)
        
    # print("run status after submitting outputs: ", run.status)
    # if run.status != "requires_action":
    bot_message = get_latest_message(client, thread, last_message)
    return bot_message

def trackEvent(email, scope, groupBy=None, analysis=True):
    url = "http://localhost:3000/track"
    headers = {'Content-Type': 'application/json'}

    data = {
        "orgId": "1",
        "email": email,
        "scope": scope,
        "groupBy": groupBy if groupBy is not None else 'event',
        "analysis": analysis if analysis is not None else True
    }

    response = requests.get(url, headers=headers, data=json.dumps(data))
    print(response.json())
    return response.json()

def createEvent(email, startTime, endTime, timezone="America/New_York", summary=None, description=None, location=None):
    url = "http://localhost:3000/manage"
    headers = {'Content-Type': 'application/json'}

    data = {
        "orgId": "1",
        "email": email,
        "type": "insert",
        "startTime": startTime,
        "endTime": endTime,
        "timezone": timezone,
        "summary": summary,
        "description": description,
        "location": location
    }

    # Remove keys with None values
    # data = {k: v for k, v in data.items() if v is not None}

    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.json())
    return response.json()

def updateEvent(email, startTime, endTime, eventId, timezone="America/New_York", summary=None, description=None, location=None):
    url = "http://localhost:3000/manage"
    headers = {'Content-Type': 'application/json'}

    data = {
        "orgId": "1",
        "email": email,
        "type": "update",
        "eventId": eventId,
        "startTime": startTime,
        "endTime": endTime,
        "timezone": timezone,
        "summary": summary,
        "description": description,
        "location": location
    }

    # Remove keys with None values
    # data = {k: v for k, v in data.items() if v is not None}

    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.json())
    return response.json()

def deleteEvent(email, eventId):
    url = "http://localhost:3000/manage"
    headers = {'Content-Type': 'application/json'}

    data = {
        "orgId": "1",
        "email": email,
        "type": "delete",
        "eventId": eventId
    }

    # Remove keys with None values
    # data = {k: v for k, v in data.items() if v is not None}

    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.json())
    return response.json()

def getAnalytics(orgId):
    url = "http://localhost:3000/analytics"
    headers = {'Content-Type': 'application/json'}

    data = {
        "orgId": "1"
    }

    response = requests.get(url, headers=headers, data=json.dumps(data))
    print(response.json())
    return response.json()

def getFreeSlots(email, scope: int = 7):
    """
    Retrieves the free time slots for a given email address within a specified scope.

    Parameters:
    - email (str): The email address for which to retrieve free time slots.
    - scope (int): The number of days to consider for free time slots. Between 0 and 30. Default is 7.

    Returns:
    - list: A list of dictionaries containing the date and free time for each slot.
    """
    
    url = "http://localhost:3000/free-slot"
    headers = {'Content-Type': 'application/json'}

    data = {
        "orgId": "1",
        "email": email,
        "scope": scope
    }

    response = requests.get(url, headers=headers, data=json.dumps(data))
    print(response.json())
    return response.json()

    # slots = response.json().get("freeSlots", [])
    # return slots

# start a new thread
def start_thread_and_add_default_msg(client, email):
    global global_thread_id
    if global_thread_id is None:
        thread = client.beta.threads.create()
        current_datetime = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=f"my email is {email}"
        )
        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=f"Current date and time is {current_datetime}"
        )
        global_thread_id = thread.id
    else:
        thread = client.beta.threads.retrieve(global_thread_id)
    return thread

def get_response(client, thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")

def pretty_print(messages):
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()

# Add a new message to the thread and start a run
def submit_message(client, thread, user_message):
    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )

    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id="asst_2x9WVVd9cMQObO4gFsMhIrbC",
        instructions="Use the approprite function tool to achieve user's specific request."
    ), message    
    
# See all messages in the thread
def see_all_messages(thread):
    pretty_print(get_response(thread))
    

def wait_on_run(client, run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(2)    
    return run

def wait_on_run_after_requires_action(client, run, thread):
    while run.status == "queued" or run.status == "in_progress" or run.status == "requires_action":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(2)    
    return run

# def wait_on_run(client, run, thread, message):
#     while run.status in ["queued", "in_progress", "requires_action"]:
#         run = client.beta.threads.runs.retrieve(
#             thread_id=thread.id,
#             run_id=run.id,
#         )
        
#         # Check for "requires_action" status and handle it
#         if run.status == "requires_action":
#             submit_outputs(client, thread, run)
#         #     run = client.beta.threads.runs.retrieve(
#         #     thread_id=thread.id,
#         #     run_id=run.id,
#         # )

#         # Only continue the loop if the status is still "queued" or "in_progress"
#         if run.status in ["queued", "in_progress"]:
#             time.sleep(2)
#             # messages = client.beta.threads.messages.list(thread_id=thread.id)
#             bot_message = get_latest_message(client, thread, message)
#             break

#         time.sleep(2)
#     return bot_message



# submit tool outputs
def submit_outputs(client, thread, run):
    tool_call = run.required_action.submit_tool_outputs.tool_calls[0]
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    responses = globals()[name](**arguments)
    # print("API responses: ", responses)
    
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
    
def get_latest_message(client, thread, message):
    messages = client.beta.threads.messages.list(
    thread_id=thread.id, order="asc", after=message.id
)
    # print("last_message: ", message)
    # print("messages: ", messages)
    # print("returned message: ", messages.data[0].content[0].text.value)
    return messages.data[0].content[0].text.value