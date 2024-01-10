# Description: A chatbot that can organise your events for you!

from configs.config import OPENAI_API_KEY, PASSWORD, USERNAME, CALENDAR_URL, CREDENTIALS_FILE_PATH, TIMEZONE
import openai
import sys
import json
from openai_decorator import openaifunc, get_openai_funcs
import dateparser
from datetime import datetime, timedelta
from dateutil import parser
from openai_decorator import openaifunc
from Calendar import Calendar
import pytz

openai.api_key = OPENAI_API_KEY


@openaifunc
def check_calendar(startDate: str = "now", endDate: str = "tomorrow"):
  """
  This function checks the calendar between two dates to see if the user is available or if they have anything planned
  @param startDate: the start of the range
  @param endDate: the end of the range
  """
  credentials_file = CREDENTIALS_FILE_PATH
  timezone = TIMEZONE
  username = USERNAME

  start_range = convert_conversation_dates_to_datetime(startDate)
  if startDate == endDate:
    end_range = start_range + timedelta(days=1)
  else:
    end_range = convert_conversation_dates_to_datetime(endDate)


  calendar = Calendar(credentials_file, username, timezone)
  events = calendar.get_calendar_events(start_range, end_range)

  if events is None or len(events) == 0:
    return "I'm free"

  returnString = "I have "
  i = 0
  while i < len(events):
    event = events[i]
    returnString += event['summary'] + " from " + event['start'] + " to " + event['end']
    i += 1
    if i != len(events):
      returnString += "and then "

  return returnString


@openaifunc
def book_event(eventSummary: str = "AutoNamed", startDate: str = "NOT SET", endDate: str = "NOT SET", eventLocation: str = ""):
  """
  When a person wants to organise a time, this function checks if the user is free and then books the event in the calendar if they are free
  @param eventSummary: a summary of the event
  @param startDate: the start of the range
  @param endDate: the end of the range
  @param eventLocation: the location where the event will be taking place
  """

  if endDate == "NOT SET" or startDate == "NOT SET":
    return "When do you want to start and finish?"
  credentials_file = CREDENTIALS_FILE_PATH
  timezone = TIMEZONE
  username = USERNAME
  calendar = Calendar(credentials_file, username, timezone)

  availability = check_calendar(str(startDate), str(endDate))
  if availability != "I'm free":
    return "Sorry, I have " + availability

  startDate = convert_conversation_dates_to_datetime(startDate)
  endDate = convert_conversation_dates_to_datetime(endDate)

  already_booked_events = calendar.get_calendar_events(startDate, endDate)
  for event in already_booked_events:
    # string_format = "%Y-%m-%d %H:%M:%S%z"
    timezone = pytz.timezone(timezone)
    formatted_event_end = parser.isoparse(event['end'])
    formatted_event_start = parser.isoparse(event['start'])
    if (formatted_event_end >= timezone.localize(startDate) and formatted_event_end <= timezone.localize(endDate)) or (formatted_event_start >= timezone.localize(startDate) and formatted_event_start <= timezone.localize(endDate)):
      return "Sorry, I have " + event['summary'] # TO DO: Add in, "but i am free at ----"

  calendar.add_event(eventSummary, startDate, endDate, eventLocation)

  return "Great, booked in for " + str(startDate) + " to " + str(endDate)

@openaifunc
def edit_event(old_summary: str = "AutoNamed", old_start: str = "NOT SET", old_end: str = "NOT SET", old_location: str = "", new_summary: str = None, new_start: str = None, new_end: str = None, new_location: str = None):
  """
  Updates an event which has already been put in the calendar. It can rename the event or change the time or place of an event. This should only be executed if the user confirms they want to change the event
  @param old_summary: the old summary of the event
  @param old_start: the old time the event started
  @param old_end: the old time the event ended
  @param old_location: the old location where the event was going to take place
  @param new_summary: an updated summary of the event
  @param new_start: the new time the event will start
  @param new_end: the new time the event will end
  @param new_location: the new location where the event will take place
  """
  credentials_file = CREDENTIALS_FILE_PATH
  timezone = TIMEZONE
  username = USERNAME
  
  calendar = Calendar(credentials_file, username, timezone)

  calendar.update_event(old_summary, convert_conversation_dates_to_datetime(old_start).replace(tzinfo=None), convert_conversation_dates_to_datetime(old_end).replace(tzinfo=None), old_location, new_summary, convert_conversation_dates_to_datetime(new_start).replace(tzinfo=None), convert_conversation_dates_to_datetime(new_end).replace(tzinfo=None), new_location)

def convert_conversation_dates_to_datetime(natural_language_date):
  parsed_date = dateparser.parse(natural_language_date)
  # print(natural_language_date)
  if parsed_date:
      return parsed_date
  else:
      raise ValueError("Invalid date")



# ChatGPT API Function
def send_message(message, messages):
    # add user message to message list
    messages.append(message)

    try:
        # send prompt to chatgpt
        response = openai.ChatCompletion.create(
            # model="gpt-4-0613",
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=get_openai_funcs(),
            function_call="auto",
        )
    except openai.error.AuthenticationError:
        print("AuthenticationError: Check your API-key")
        sys.exit(1)

    # add response to message list
    messages.append(response["choices"][0]["message"])

    return messages


# MAIN FUNCTION
def run_conversation(prompt, messages=[]):
    # add user prompt to chatgpt messages
    messages = send_message({"role": "user", "content": prompt}, messages)

    # get chatgpt response
    message = messages[-1]

    # loop until project is finished
    while True:
        if message.get("function_call"):
            # get function name and arguments
            function_name = message["function_call"]["name"]
            arguments = json.loads(message["function_call"]["arguments"])

            # call function dangerously
            function_response = globals()[function_name](**arguments)

            # send function result to chatgpt
            messages = send_message(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
                messages,
            )
        else:
            # if chatgpt doesn't respond with a function call, ask user for input
            print("ChatGPT: " + message["content"])

            user_message = input("You: ")

            # send user message to chatgpt
            messages = send_message(
                {
                    "role": "user",
                    "content": user_message,
                },
                messages,
            )

        # save last response for the while loop
        message = messages[-1]

# ASK FOR PROMPT
print(
    "I'm just a chatbot, but I can also organise your events for you!"
)
prompt = input("You: ")

# RUN CONVERSATION
run_conversation(prompt)