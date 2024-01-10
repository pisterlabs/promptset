import re
import datetime
from dateutil import parser
import os.path
import pytz
from tzlocal import get_localzone
import openai
with open('openai.txt') as f:
    openai.api_key = f.read()
import json
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import requests
SCOPES = ['https://www.googleapis.com/auth/calendar.events']

def readable_time(time, format):
    formatted_time = datetime.datetime.strptime(time, format)
    day = formatted_time.day
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    new_format = f"%A the {day}{suffix} of %B, %Y at %H:%M"
    new_time = datetime.datetime.strftime(formatted_time,new_format)
    return new_time

def describe_weather(weathercode):
    weather_descriptions = {
        0: "Clear sky ☀️",
        1: "Mostly clear 🌤",
        2: "Partly cloudy ⛅",
        3: "Overcast ☁️",
        45: "Fog 🌫",
        48: "Dense fog 🌁",
        51: "Light drizzle 🌦",
        53: "Drizzle 🌧",
        55: "Heavy drizzle 🌧️🌧️",
        56: "Light freezing drizzle 🌨️",
        57: "Freezing drizzle ❄️💧",
        61: "Light rain 🌦",
        63: "Rain 🌧",
        65: "Heavy rain 🌧️🌧️",
        66: "Light freezing rain 🌨️",
        67: "Freezing rain ❄️💧",
        71: "Light sleet 🌦❄️",
        73: "Sleet 🌨️",
        75: "Heavy sleet 🌨️🌨️",
        81: "Light snowfall 🌨️",
        83: "Snowfall ❄️",
        85: "Heavy snowfall ❄️❄️"
    }
    return weather_descriptions.get(weathercode, f"Unknown weather code {weathercode} ❓")



def get_weather(date=None):
    base_url = "https://api.open-meteo.com/v1/forecast"
    if date is None:
        date = (datetime.now()).strftime("%Y-%m-%dT%H:00")
    else:
        date = date[:-5]+"00"
    print(date)
    complete_url = f"{base_url}?latitude={lat}&longitude={lon}&hourly=temperature_2m,weathercode&time={date}"

    try:
        response = requests.get(complete_url)
        data = response.json()
        weather_info = None
        for i, time in enumerate(data['hourly']['time']):
            if not time == date:
                continue
            print("Found it!")
            weather_info = {
                "description": describe_weather(data['hourly']["weathercode"][i]),
                "temperature": data['hourly']["temperature_2m"][i],
            }
            break
        return weather_info
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


city_name = "Copenhagen"
country_code = "Denmark"

complete_url = f"https://nominatim.openstreetmap.org/search?city={city_name}&country={country_code}&format=json"
response = requests.get(complete_url)
data = response.json()
lat = float(data[0]["lat"])
lon = float(data[0]["lon"])

class ScryException(BaseException):
    pass

class RegretException(BaseException):
    pass

class GetNextEventException(BaseException):
    def __init__(self, num_events):
        self.num_events = num_events


class GetEventsBetweenTimesException(BaseException):
    def __init__(self, start_time,end_time):
        self.start_time = start_time
        self.end_time = end_time

def get_now():
    timezone = pytz.timezone('Etc/GMT-2')
    now = datetime.datetime.now(timezone)
    return now

def delete_events(service, event_ids):
    for event_id in event_ids:
        event = service.events().get(calendarId='primary', eventId=event_id).execute()
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        print(f"Event named '{event['summary']}' deleted successfully.")


def update_events(service, event_ids, revised_events):
    yes_to_all = False
    for event_id, revised_event in zip(event_ids, revised_events):
        event = service.events().get(calendarId='primary', eventId=event_id).execute()
        changes = []
        if revised_event['summary'] and event['summary'] != revised_event['summary']:
            changes.append(
                f"Summary: {event['summary']} -> {revised_event['summary']}\n")
        if revised_event['description'] and event['description'] != revised_event['description']:
            changes.append(
                f"Description: {event['description']} -> {revised_event['description']}\n")
        if revised_event['start'] and event['start']['dateTime'] != revised_event['start']:
            readable_old_time = readable_time(event['start']['dateTime'],"%Y-%m-%dT%H:%M:%S")
            readable_new_time = readable_time(
                revised_event['start'], "%Y-%m-%dT%H:%M:%S")
            changes.append(f"Start time: {readable_old_time} -> {readable_new_time}\n")
        if revised_event['end'] and event['end']['dateTime'] != revised_event['end']:
            readable_old_time = readable_time(event['end']['dateTime'],"%Y-%m-%dT%H:%M:%S")
            readable_new_time = readable_time(
                revised_event['end'], "%Y-%m-%dT%H:%M:%S")
            changes.append(
                f"End time: {readable_old_time} -> {readable_new_time}\n")
        if revised_event['reminders'] and event['reminders'] != revised_event['reminders']:
            changes.append(
                f"Reminders: {event['reminders']} -> {revised_event['reminders']}\n")
        if changes:
            if not yes_to_all:
                print()
                print(f"These changes are about to be made to \"{event['summary']}\":\n     - " +
                        "     - ".join(changes))
                confirm = get_input("Confirm changes?", "yes")
                if confirm.lower() in ["no", "n"]:
                    continue
                if confirm.lower() in ["yy", "yes to all"]:
                    if get_input("This will approve all edits, even ones you've disapprove. Write \"yes\" to affirm.") == "yes":
                        yes_to_all = True
                if confirm.lower() in ["nn", "no to all"]:
                    if get_input("This will disapprove all edits, even ones you've approved. Write \"yes\" to affirm.") == "yes":
                        return
        else:
            print("No changes to be made for event '",event['summary'],"'")
            continue

        event['summary'] = revised_event['summary']
        event['description'] = revised_event['description']
        event['start']['dateTime'] = revised_event['start']
        event['end']['dateTime'] = revised_event['end']
        event['reminders'] = revised_event['reminders']

        updated_event = service.events().update(
            calendarId='primary',
            eventId=event_id,
            body=event
        ).execute()

        print(f"Event '{updated_event['summary']}' updated successfully.")
    print()

def get_events_between_times(service, start_time=None, end_time=None):
    if start_time is None:
        start_time = get_now().isoformat()
    if end_time is None:
        end_time = get_now().replace(hour=23, minute=59, second=59).isoformat()
    
    events_result = service.events().list(
        calendarId="primary", timeMin=start_time, timeMax=end_time).execute()
    events = events_result.get("items", [])

    event_list = []
    print()

    if not events:
        print("No events found.")
        return None
    else:
        print("Found events:")
        for event in events:
            event_id = event["id"]
            start = event["start"].get("dateTime", event["start"].get("date"))
            end = event["end"].get("dateTime", event["end"].get("date"))
            summary = event["summary"]
            description = event.get("description", "")
            reminders = event.get("reminders", {})
            print(start, summary)
            event_list.append({
                "id": event_id,
                "start": start,
                "end": end,
                "summary": summary,
                "description": description,
                "reminders": reminders
            })
    json_output = json.dumps(event_list)
    return json_output

def get_next_event(service,amount=1):
    if amount <= 0:
        return None
    now = get_now().isoformat()
    print(f'Getting the upcoming {amount} events')
    events_result = service.events().list(calendarId='primary', timeMin=now,
                                          maxResults=amount, singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])

    if not events:
        print('No upcoming events found.')
        return

    for event in events:
        print(event)
        start = event['start'].get('dateTime', event['start'].get('date'))
        print(start, event['summary'])
    return events

def ask_the_bot(question, context=[], temperature=0.05, bot = 'gpt-3.5-turbo'):
    context.append({"role": "user", "content": question})
    max_resends = 2
    request_resends = 0
    while request_resends <= max_resends:
        try:
            response = openai.ChatCompletion.create(
                model=bot,
                messages=context,
                temperature=temperature
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print()
            print(f"An error occured. Resending ({request_resends}/{max_resends})...")
            request_resends = request_resends + 1
            time.sleep(5)
    print("\nAll requests failed! Error: ",e)
    return []

def to_message(message,role):
    return {"content": message, "role": role}

def converse_the_bot(context, temperature=0.2, bot='gpt-3.5-turbo'):
    max_resends = 2
    request_resends = 0
    while request_resends <= max_resends:
        try:
            response = openai.ChatCompletion.create(
                model=bot,
                messages=context,
                temperature=temperature
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)
            print(f"An error occured. Resending ({request_resends}/{max_resends})...")
            request_resends = request_resends + 1
            time.sleep(5)
    print("\nAll requests failed!")
    return []

def discuss_until_ok(prompt_to_bot,bot='gpt-3.5-turbo',temperature = 0.2):
    print(f"User: {prompt_to_bot}\n")
    context = []
    context.append(to_message(prompt_to_bot,"user"))
    while True:
        from_bot = converse_the_bot(context,bot=bot, temperature=temperature)
        context.append(to_message(from_bot,"assistant"))
        print(f"Bot response: {from_bot}\n")
        
        to_bot = get_input("Ok?","This is okay.")
        context.append(to_message(to_bot,"user"))

        if to_bot == "This is okay.":
            break

        # Reset to gpt-3.5 for cost
        bot='gpt-3.5-turbo'
    print()
    return from_bot

def delete_events_from_string(service, planning_prompt):
    events = json.loads(events_from_prompt(service,planning_prompt))
    for e in events:
        e.pop('description')
        e.pop('reminders')
    events = json.dumps(events)

    today = datetime.datetime.now().strftime('%Y-%m-%d, %A')
    time = datetime.datetime.now().strftime('%H:%M')
    if events:
        delete_events_prompt = f"From this query: \"{planning_prompt} [sent {today}, {time}]\", do as follows: 1. Identify the intent of the query. 2. Pick out the ids of any events from the JSON array below that are described by the intent or query. 3. Make an array consisting of just the ids of the to-be deleted events correlating to the titles/summaries from step 2. JSON Array: {events}."
        delete_events_response = discuss_until_ok(delete_events_prompt)
        deletable_ids = try_to_load_json_from_string(delete_events_response)
    
        for id in deletable_ids:
            approved_ids = []
            event = service.events().get(calendarId='primary', eventId=id).execute()
            print(f"You are about to delete event: {event['summary']} from your calendar.")
            approval = get_input("Confirm changes?", "no")
            if approval == 'no' or approval == 'n':
                continue
            elif ((approval == 'no to all' or approval == 'nn') and get_input('This will delete no events, even ones you have approved for deletion. Write \'yes\' to confirm.') == 'yes'):
                return []
            elif (approval == 'yes to all' or approval == 'yy') and get_input('This will delete all the detected events, even ones you have disapproved for deletion. Write \'yes\' to confirm') == 'yes':
                approved_ids = deletable_ids
                break
            else:
                approved_ids.append(id)

        delete_events(service, approved_ids)
    return []

def generate_events_from_context(service, planning_prompt):
    events = events_from_prompt(service, planning_prompt)

    local_tz = get_localzone()
    current_datetime = datetime.datetime.now(local_tz)
    day = current_datetime.strftime('%A')
    today = current_datetime.date().isoformat()
    time = current_datetime.time().strftime('%H:%M:%S')
    tz_offset = current_datetime.strftime('%z')[:3]+":"+current_datetime.strftime('%z')[3:]

    if not events:
        event_string = "no plans"
    else:
        events = try_to_load_json_from_string(events)
        event_strings = [f"{event['summary']}, starting at {event['start']}, ending at {event['end']}" for event in events]
        event_string = ", ".join(event_strings)

    new_events_prompt = f"Given the following current date and time: {day}, {today}T{time}:00, the following sequence of preexisting plans: {event_string} and planning prompt: '{planning_prompt}', do the following. 1. Identify the intent of the prompt. 2. Find any completely free time windows between the preexisting plans to use for new events. If none exists, after the events is fine. 3. In detail, list your intended additions with respect to the query. 4. Make a new JSON array consisting of just the additions with the following keys: summary, start_datetime,  end_datetime, description, and reminder (int, minutes), in an array that can be parsed to create calendar events. Please use 1-2 emojis per complex sentence in the title and description to make them more personal."

    print()
    new_events_response = discuss_until_ok(new_events_prompt)
    events_json = try_to_load_json_from_string(new_events_response)
    approved_events = []
    for event in events_json:
        print()
        print(f"Add event: {event['summary']} to calendar?")
        approval = get_input("Confirm changes?", "yes")
        if approval == 'no' or approval == 'n':
            continue
        elif ((approval == 'no to all' or approval == 'nn') and get_input('This will add no new events, even ones you have approved to the calender. Confirm?', 'yes') == 'yes'):
            return []
        elif (approval == 'yes to all' or approval == 'yy') and get_input('This will add all new events, even ones you have disapproved to the calender. Confirm?', 'yes') == 'yes':
            return events_json
        else:
            approved_events.append(event)
    return approved_events


def time_window_from_prompt(planning_prompt):
    today = datetime.datetime.now().strftime('%Y-%m-%d, %A')
    time = datetime.datetime.now().strftime('%H:%M')
    time_window_prompt = f"From this query: \"{planning_prompt} [sent {today}, {time}]\", do as follows: 1. Identify the intent of the query. 2. Explain in depth the most important times mentioned in the query. If no day information is present in the query, assume today. If no temporal hints are present in the query (other than the query time stamp), simply start at 00:00 and end at 23:59 of the same day. 3. End your response with an unambiguous time frame that covers the original/current plans from step 2. Fill out the following completely with no changes to the format: 'Original plans: YYYY-MM-DD HH:MM to YYYY-MM-DD HH:MM'."

    print()
    time_window_response = discuss_until_ok(time_window_prompt)

    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} to \d{4}-\d{2}-\d{2} \d{2}:\d{2}"
    match = re.search(pattern, time_window_response)

    if match:
        time_window = match.group()
        start_time, end_time = time_window.split(" to ")
    else:
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        start_time = today + " 00:00"
        end_time = today + " 23:59"
        print("No time window found in the response. Assuming the entire present day.")
        confirmation = input("Is the assumption correct? (yes/no): ")
        if confirmation.lower() != "yes":
            start_time = get_input("Please provide the correct start time.",default_value=start_time)
            end_time = get_input("Please provide the correct end time.",default_value=end_time)

    start_time = parse_date_time(start_time).replace(second=0).isoformat()
    end_time = parse_date_time(end_time).replace(second=0).isoformat()
    return start_time, end_time

def events_from_prompt(service,planning_prompt):
    print()
    print("Identifying time window from prompt...")
    start_time, end_time = time_window_from_prompt(planning_prompt)
    print(
        f"Fetching events between {readable_time(start_time, '%Y-%m-%dT%H:%M:%S%z')} and {readable_time(start_time, '%Y-%m-%dT%H:%M:%S%z')}.")
    events = get_events_between_times(service,start_time,end_time)
    return events

def update_events_from_string(service,planning_prompt):
    events = events_from_prompt(service,planning_prompt)
    today = datetime.datetime.now().strftime('%Y-%m-%d, %A')
    time = datetime.datetime.now().strftime('%H:%M')
    if events:
        update_events_prompt = f"From this query: \"{planning_prompt} [sent {today}, {time}]\", do as follows: 1. Identify the intent of the query. 2. Pick out the titles/summaries of any events from the JSON array below that are discussed in the query. 3. In detail, list your intended edits to the events with respect to the query. 4. Make a new JSON array consisting of just the now revised events correlating to the titles/summaries from step 2. Make sure to be unambiguous, autonomously and intelligently making any decisions necessary to satisfy the query, and escape any special characters that have special meaning in JSON by putting backslash before it. JSON Array: {events}."
        print()
        update_events_response = discuss_until_ok(update_events_prompt)
        updated_events = try_to_load_json_from_string(update_events_response)
        event_ids = [obj['id'] for obj in updated_events]
        revised_events = [{k: v for k, v in obj.items() if k != 'id'} for obj in updated_events]
        update_events(service,event_ids,revised_events)
    return []

def try_to_load_json_from_string(json_string):
    print()
    print("Trying to load JSON from GPT.")

    start_of_json = json_string.find('{')
    end_of_json = json_string.rfind('}')+1
    json_string = json_string[start_of_json:end_of_json]
    json_string = "["+json_string+"]"
    
    try:
        loaded_json = json.loads(json_string)
    except json.JSONDecodeError as e:
        print("Invalid JSON, trying to fix...")
        fixed_json_prompt = "This JSON array" + json_string + \
            "\n\ngives this error:\n" + str(e) + "\nplease fix it."
        print(f"User:\n{fixed_json_prompt}")
        fixed_events_response = ask_the_bot(fixed_json_prompt, [])
        print()
        print(f"Bot response:\n{fixed_events_response}")
        start_of_json = fixed_events_response.find('[')
        end_of_json = fixed_events_response.rfind(']')+1
        json_string = fixed_events_response[start_of_json:end_of_json]
        try:
            loaded_json = json.loads(json_string)
        except json.JSONDecodeError:
            print()
            print("Failed to get a valid response from the GPT.\n")
            return []
    print("Loading JSON succeeded!")
    return loaded_json

def generate_events_from_string(service, planning_prompt):
    local_tz = get_localzone()
    current_datetime = datetime.datetime.now(local_tz)
    day = current_datetime.strftime('%A')
    today = current_datetime.date().isoformat()
    time = current_datetime.time().strftime('%H:%M:%S')
    tz_offset = current_datetime.strftime('%z')[:3]+":"+current_datetime.strftime('%z')[3:]

    new_events_prompt = f"Given the following current date and time: {day}, {today}T{time} and planning prompt: '{planning_prompt}', format the prompt's contents as JSON objects with the following keys: summary, start_datetime,  end_datetime, description, and reminder (int, minutes), in an array that can be parsed to create calendar events. Please use 1-2 emojis per complex sentence in the title and description to make them more personal."
    while not planning_prompt:
        planning_prompt = get_input("Please enter concrete events")
        new_events_prompt = f"Given the following current date and time: {day}, {today}T{time}:00 and events: '{planning_prompt}', format them as JSON objects with the following keys: summary, start_datetime,  end_datetime, description, and reminder (int, minutes), in an array that can be parsed to create calendar events. Please use 1-2 emojis per complex sentence in the title and description to make them more personal."
    print()
    new_events_response = discuss_until_ok(new_events_prompt,bot='gpt-4-1106-preview',temperature=1)
    events_json = try_to_load_json_from_string(new_events_response)
    approved_events = []
    for event in events_json:
        print()
        print(f"Add event: {event['summary']} to calendar?")
        approval = get_input("Confirm changes?", "yes")
        if approval == 'no' or approval == 'n':
            continue
        elif ((approval == 'no to all' or approval == 'nn') and get_input('This will add no new events, even ones you have approved to the calender. Confirm?', 'yes') == 'yes'):
            return []
        elif (approval == 'yes to all' or approval == 'yy') and get_input('This will add all new events, even ones you have disapproved to the calender. Confirm?', 'yes') == 'yes':
            return events_json
        else:
            approved_events.append(event)
    return approved_events

def completion_from_string(service, planning_prompt):
    events = json.loads(events_from_prompt(service,planning_prompt))
    unedited_events = [dict(e) for e in events]
    for e in events:
        e.pop('description')
        e.pop('reminders')
    events = json.dumps(events)
    today = datetime.datetime.now().strftime('%Y-%m-%d, %A')
    time = datetime.datetime.now().strftime('%H:%M')
    if events:
        update_events_prompt = f"From this query: \"{planning_prompt} [sent {today}, {time}]\", do as follows: 1. Pick out the titles/summaries of any events from the JSON array below that are discussed in the query. 2. Make a new JSON array consisting of just the events correlating to the titles/summaries from step 1. now with a green checkmark emoji (✅) prepended onto the summary. The emoji shouldn't replace or remove other emojis present and should be on the far left of it. Make sure to be unambiguous and escape any special characters that have special meaning in JSON by putting backslash before it. JSON Array: {events}."
        print()
        update_events_response = discuss_until_ok(update_events_prompt)
        updated_events = try_to_load_json_from_string(update_events_response)
        event_ids = [obj['id'] for obj in updated_events]
        revised_events = [{k: v for k, v in obj.items() if k != 'id'} for obj in updated_events]
        for e, u in zip(revised_events,unedited_events):
            e['description'] = u['description']
            e['reminders'] = u['reminders']
        update_events(service,event_ids,revised_events)
    return []

def multiquery_from_string(service, planning_prompt):
    commands_string = ""
    invalid_commands = ["SEQUENTIALLY"]
    for command in COMMANDS:
        if not command['name'] in invalid_commands:
            commands_string += f"{command['name']}, {command['description']}"
    break_down_prompt = f"Only respond in JSON array code block. Given the following query \"{planning_prompt}\", do the following:  1. From the perspective of a calendar app, identify the intents of the query. 2. Identify the most suitable commands to satisfy the query use from these: {commands_string} Make an array matching the intent of each subpart of the query using a key named \"subquery\" and the chosen command for that subquery with a key named \"command\" and return a JSON array of this."
    
    print()
    break_down_prompt_response = discuss_until_ok(break_down_prompt)
    subqueries_and_commands = try_to_load_json_from_string(break_down_prompt_response)
    print()

    new_events = []
    for action in subqueries_and_commands:
        action['subquery'] = f"I need just this part done: {action['subquery']}, from this list: {planning_prompt}. Everything before this in the list has been done already."
        print(f"\nBot response:\nAlright, we'll {action['command']} plans!")
        for command in COMMANDS:
            if action['command'] == command['name']:
                command_function = command['command']
        events = command_function(service, action['subquery'])
        new_events.append(events)
        print("Waiting 5 seconds to do next task.")
        time.sleep(5)
    return new_events

def events_from_paste(service, planning_prompt):
    events_json = try_to_load_json_from_string(planning_prompt)
    approved_events = []
    for event in events_json:
        print()
        print(f"Add event: {event['summary']} to calendar?")
        approval = get_input("Confirm changes?", "yes")
        if approval == 'no' or approval == 'n':
            continue
        elif ((approval == 'no to all' or approval == 'nn') and get_input('This will add no new events, even ones you have approved to the calender. Confirm?', 'yes') == 'yes'):
            return []
        elif (approval == 'yes to all' or approval == 'yy') and get_input('This will add all new events, even ones you have disapproved to the calender. Confirm?', 'yes') == 'yes':
            return events_json
        else:
            approved_events.append(event)
    print(approved_events)
    return approved_events

def print_help(service,planning_prompt):
    command_listing = "\n"
    for command in COMMANDS:
        command_listing += f"{command['name']} - {command['description']}\n"
    print(command_listing)
    return []

def manual_planning_main(service, planning_prompt=""):
    choice = get_input("What would you like to do? Type help for list of commands.", "make").upper()

    chosen_command = None
    for command in COMMANDS:
        if command['name'] == choice:
            chosen_command = command
            break

    if chosen_command:
        print(f"\nBot response:\nAlright, we'll {chosen_command['name']} plans!")
        if not choice == "HELP":
            planning_prompt = get_input("Please enter a general plan", None)
        return chosen_command['command'](service, planning_prompt)
    else:
        print("\nBot response:\nSorry, I couldn't understand your response.")

def intelligent_planning_main(service, planning_prompt=""):
    planning_prompt = get_input("Please enter a general plan", None)
    if not planning_prompt:
        return generate_events_from_string(service,"")

    command_choosing_prompt = f"Given the following query \"{planning_prompt}\", do the following:  1. From the perspective of a calendar app, identify the intent of the query. 2. Identify the most suitable command to satisfy the query use from these:"
    for command in COMMANDS:
        command_choosing_prompt += f" {command['name']}, {command['description']}"
    command_choosing_prompt += " Make sure only to include one mention of an existing command in your response."
    command_choosing_response = discuss_until_ok(command_choosing_prompt)

    chosen_command = None
    for command in COMMANDS:
        if command['name'] in command_choosing_response:
            chosen_command = command
            break

    if chosen_command:
        print(f"\nBot response:\nAlright, we'll {chosen_command['name']} plans!")
        return chosen_command['command'](service, planning_prompt)
    else:
        print("\nBot response:\nSorry, I couldn't understand your response.")

COMMANDS = [{
        "name": "MAKE",
        "description": "makes a new event from prompt.",
        "command": generate_events_from_string
}, {
    "name": "EDIT",
    "description": "finds and edits calendar events.",
    "command": update_events_from_string
}, {
        "name": "DELETE",
        "description": "removes events from the calendar.",
        "command": delete_events_from_string
}, {
        "name": "COMPLETE",
        "description": "marks events as complete or done.",
        "command": completion_from_string
}, {
        "name": "GET THEN MAKE",
        "description": "intelligently makes events.",
        "command": generate_events_from_context
}, {
        "name": "HELP",
        "description": "displays the list of possible commands.",
        "command": print_help
}, {
        "name": "REGRET",
        "description": "undos the current command.",
        "command": None
}, {
        "name": "PASTE",
        "description": "Makes events from an already processed request.",
        "command": events_from_paste
}, {
    "name": "EXIT",
    "description": "exits CalendarMe.",
    "command": None
}]

def parse_time(time_str):
    return datetime.datetime.strptime(time_str, "%H:%M").time()

def parse_minutes(minutes_str):
    minutes = int(minutes_str)
    return datetime.timedelta(minutes=minutes)


def parse_date_time(date_time_str, timezone_str='Etc/GMT-2'):
    timezone = pytz.timezone(timezone_str)
    date_time = parser.parse(date_time_str)
    date_time = timezone.localize(date_time)
    return date_time

def calculate_time_with_delta(base_time, delta):
    base_datetime = datetime.datetime.combine(datetime.date.today(), base_time)
    result_datetime = base_datetime + delta
    return result_datetime.time()

def scry(service):
    now = get_now().isoformat()
    print('Getting the upcoming 5 events')
    events_result = service.events().list(calendarId='primary', timeMin=now,
                                            maxResults=5, singleEvents=True,
                                            orderBy='startTime').execute()
    events = events_result.get('items', [])

    print()
    if not events:
        print('No upcoming events found.')
        return

    # Prints the start and name of the next 10 events
    for event in events:
        print(event)
        start = event['start'].get('dateTime', event['start'].get('date'))
        print(start, event['summary'])


def create_event(service, event_title, start_datetime, end_datetime, description="No description", reminder=None):
    event = {
        'summary': event_title,
        'description': description,
        'start': {
            'dateTime': start_datetime,
            'timeZone': 'Etc/GMT-2',
        },
        'end': {
            'dateTime': end_datetime,
            'timeZone': 'Etc/GMT-2',
        },
    }

    if reminder is not None:
        event['reminders'] = {
            'useDefault': False,
            'overrides': reminder
        }
    else: 
        event['reminders'] = {
            'useDefault': False,
            'overrides': []
        }
    event = service.events().insert(calendarId='primary', body=event).execute()
    print('Event created: %s' % (event.get('htmlLink')))

def parse_time_input(time_input, default_time):
    if time_input.startswith("+"):
        minutes = int(time_input[1:])
        time = calculate_time_with_delta(
            default_time, datetime.timedelta(minutes=minutes)).time()
    else:
        time = datetime.datetime.strptime(time_input, '%H:%M').time()
    return time

def get_input(msg,default_value=None,default_msg=""):
    if not default_value == None:
        default_msg = f" (Default: {str(default_value)+default_msg})"
    user_input = input(msg+default_msg+": ") or default_value
    if user_input == "" or user_input == None:
        return
    if user_input == "exit" or user_input == "quit":
        exit()
    if user_input == 'scry':
        raise ScryException()
    if user_input == 'regret':
        raise RegretException()
    if user_input.startswith('next'):
        try:
            num_events = int(user_input[len("next"):].strip())
        except ValueError:
            num_events = 1
        raise GetNextEventException(num_events)
    if user_input.startswith('between'):
        try:
            start_time_str = user_input[len("between"):].strip()[:16]
            end_time_str = user_input[len("between"):].strip()[17:]
            start_time = parse_date_time(start_time_str).replace(second=0).isoformat()
            end_time = parse_date_time(end_time_str).replace(second=0).isoformat()
        except ValueError:
            print('Invalid date/time format. Please try again.')
            start_time, end_time = None, None
        raise GetEventsBetweenTimesException(start_time, end_time)
    return user_input



if __name__ == '__main__':
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    events = []
    try:
        service = build('calendar', 'v3', credentials=creds)
        print("Welcome to CalendarMe!")
        print("----------------------------")
        while True:
            try:
                events_json = manual_planning_main(service)

                if events_json is not None:
                    for event in events_json:
                        reminder_input = event.get('reminder')
                        reminder = [{'method': 'popup', 'minutes': int(
                            reminder_input)}] if reminder_input else None
                        weather = get_weather(event['start_datetime'])
                        weather_string = ""
                        if weather is not None:
                            weather_string = f"\n\nWeather ({readable_time(event['start_datetime'],'%Y-%m-%dT%H:%M:%S')}):\nDescription: {weather['description']}\nTemperature: {weather['temperature']}°C"
                            print(f"Weather details added to description{weather_string}")
                            creds = None
                        """Shows basic usage of the Google Calendar API.
                        Prints the start and name of the next 10 events on the user's calendar.
                        """
                        creds = None

                        if os.path.exists('token.json'):
                            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
                        if not creds or not creds.valid:
                            if creds and creds.expired and creds.refresh_token:
                                creds.refresh(Request())
                            else:
                                flow = InstalledAppFlow.from_client_secrets_file(
                                    'credentials.json', SCOPES)
                                creds = flow.run_local_server(port=0)
                            with open('token.json', 'w') as token:
                                token.write(creds.to_json())
                            with open('token.json', 'w') as token:
                                token.write(creds.to_json())
                            service = build('calendar', 'v3', credentials=creds)
                        
                        event_end_time = datetime.datetime.strptime(event['end_datetime'].split('T')[0] + " " + event['end_datetime'].split('T')[1][:8], '%Y-%m-%d %H:%M:%S')
                        time_now = datetime.datetime.now()

                        if event_end_time <= time_now:
                            event['summary'] = "✅" + event['summary']

                        create_event(service, event['summary'], event['start_datetime']+"+01:00",
                                     event['end_datetime']+"+01:00", event['description']+weather_string, reminder)
            except ScryException:
                scry(service)
            except RegretException:
                print("Let's try again!")
            except GetNextEventException as e:
                num_events = e.num_events
                get_next_event(service, num_events)
            except GetEventsBetweenTimesException as e:
                start_time = e.start_time
                end_time = e.end_time
                get_events_between_times(service, start_time, end_time)

    except HttpError as error:
        print('An error occurred: %s' % error)
