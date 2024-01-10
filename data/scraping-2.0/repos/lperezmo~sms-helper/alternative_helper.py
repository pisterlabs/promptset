import os
import json
import logging
import requests
from openai import OpenAI
import azure.functions as func
from twilio.rest import Client

#------------------------------------#
# Load environment variables
#------------------------------------#
ACCOUNT_SID = os.environ["ACCOUNT_SID"]
AUTH_TOKEN = os.environ["AUTH_TOKEN"]

#------------------------------------#
# OpenAI and Twilio Clients
#------------------------------------#
ai_client = OpenAI()
CLIENT = Client(ACCOUNT_SID, AUTH_TOKEN)

#------------------------------------#
# Security check
#------------------------------------#
def check_pin_and_reply(PIN, incoming_message):
    """
    Generate a reply based on the incoming message.
    
    Parameters
    ----------
    PIN : str
        Security PIN
    incoming_message : str
        Incoming message from Twilio
    
    Returns
    -------
    message : str
        Reply message
    """
    if incoming_message.strip() == PIN:
        return """Welcome back lord overlord Luis.
    - I can schedule calls, texts, and reminders for you.
    - I can also just answer questions or anything like that.
    - Text 'yolo' to see this message again"""
    else:
        messages = CLIENT.messages.list(from_=send_to, to=send_from)
        sent_pin = False
        for message in messages:
            if message.body.strip() == PIN:
                sent_pin = True
        if sent_pin:
            follow_up_reply = get_follow_up_text(incoming_message)
            return follow_up_reply
        else:
            return "Please provide security PIN to continue"

#------------------------------------#
# Current time
#------------------------------------#
def get_time():
    """
    Robustly get the current time from an API.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    str
        Current time
    """
    max_retries = 3
    attempts = 0
    while attempts < max_retries:
        try:
            response = requests.get('http://worldtimeapi.org/api/timezone/America/Los_Angeles')
            response.raise_for_status()  # This will raise an exception for HTTP error codes

            res = response.json()
            datetime = res.get('datetime')
            abbreviation = res.get('abbreviation')
            day_of_week = res.get('day_of_week')

            if datetime and abbreviation and day_of_week is not None:
                return f"{datetime} {abbreviation} day of the week {day_of_week}"
            else:
                raise ValueError("Incomplete time data received")
            
        except (requests.RequestException, ValueError) as e:
            attempts += 1
            if attempts == max_retries:
                return "Failed to get time after several attempts."

#-----------------------------------------#
# Generate JSON body to schedule reminder
#-----------------------------------------#
def schedule_reminder(natural_language_request):
    """
    Generate JSON body to schedule reminder
    
    Parameters
    ----------
    natural_language_request : str
        Natural language request from user
    
    Returns
    -------
    JSON body to schedule reminder
    """
    sys_prompt = """Your job is to create the JSON body for an API call to schedule texts and calls. Then , you will schedule the text or call for the user will request based on pacific time (given in pacific time). If user asks for a reminder today at 6 pm that is 18:00 (24 hour notation).

    If the user requests to be called or messaged on their work phone, set to_phone variable to '+12221110000' else send it to default phone '+15554443333'. Use twilio = True by default.

    Example endpoint: http://YOUR-ENDPOINT.elasticbeanstalk.com/schedule_single_reminder
    Example call:
    {
    "time": "18:20",
    "day": "2023-11-27",
    "message_body": "This is the reminder body!",
    "call": "True",
    "twilio": "True",
    "to_number": "+15554443333"
    }

    Example message:
    {
        "time":"23:46",
        "day":"2023-11-27",
        "message_body":"text reminder to check email",
        "to_number":"+15554443333",
        "twilio":"True",
        "call":"False"
    }
    """
    curr_time = get_time()
    ai_client = OpenAI()
    completion = ai_client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": f"{sys_prompt}"},
                {"role": "user", "content": f"{natural_language_request}. <Current Time>: {curr_time}"},
            ],
            response_format={ "type": "json_object" },
        )
    
    return json.loads(completion.choices[0].message.content)

#------------------------------------#
# Follow up text
#------------------------------------#
def get_follow_up_text(send_to, send_from, incoming_message):
    """Send follow up text
    
    Parameters
    ----------
    send_to : str
        Phone number to send text to
    send_from : str
        Phone number to send text from
    incoming_message : str
        Incoming message from Twilio
        
    Returns
    -------
    message : str
        Response from the AI to the user
    """
    if incoming_message == 'yolo':
        return """Welcome back lord overlord Luis.
        - I can schedule calls, texts, and reminders for you.
        - I can also just answer questions or anything like that.
        - Text 'yolo' to see this message again"""
    else:
        tools = [
                    {
                        "type": "function",
                        "function": {
                        "name": "schedule_reminder",
                        "description": "Schedule a reminder using natural language",
                        "parameters": {
                            "type": "object",
                            "properties": {
                            "natural_language_request": {
                                "type": "string",
                                "description": "Requested reminder in natural language. Example: 'Remind me to call mom tomorrow at 6pm' or 'Send me a message with a Matrix quote on wednesday at 8am'",
                            }
                            },
                            "required": ["natural_language_request"],
                        },
                        }
                    }
                ]
        #----------------------------------------------------#
        # AI w/tools - reply or use tools to schedule reminder
        #----------------------------------------------------#
        completion = ai_client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": f"You are an AI assistant that can schedule reminders (like calls and texts) if asked to do so. Be informative, funny, and helpful, and keep your messages clear and short. To schedule reminder just pass a natural language request to the function 'schedule_reminder'"},
                {"role": "user", "content": f"{incoming_message}"}
            ],
            tools=tools,
            tool_choice="auto"
        )
        message = completion.choices[0].message.content
        if message==None:
            message = "Just a minute while I schedule your reminder."
        else:
            return message
        #----------------------------------------------------#
        # If tools are called, call the tools function
        #----------------------------------------------------#
        if completion.choices[0].message.tool_calls:
            if completion.choices[0].message.tool_calls[0].function.name == 'schedule_reminder':
                args = completion.choices[0].message.tool_calls[0].function.arguments
                args_dict = json.loads(args)
                try:
                    #--------------------------------#
                    # Schedule reminder
                    #--------------------------------#
                    json_body = schedule_reminder(**args_dict)
                    url_endpoint = "http://YOUR-ENDPOINT.elasticbeanstalk.com/schedule_single_reminder"
                    headers = {'Content-Type': 'application/json'}
                    response = requests.post(url_endpoint, headers=headers, data=json.dumps(json_body))
                    if response.status_code == 200:
                        return "Your reminder has been scheduled."
                except Exception as e:
                    logging.error(f"Error: {e}")
                    return "Error scheduling reminder."
