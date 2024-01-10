# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions




import requests
from rasa_sdk.events import AllSlotsReset
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import UserUtteranceReverted

import datetime
from datetime import datetime, timedelta
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pickle

from Adafruit_IO import Client, Data


class ActionToggleLight(Action):
    def name(self) -> Text:
        return "action_toggle_light"

    def run(self, dispatcher: CollectingDispatcher,
        tracker: Tracker,domain: Dict[Text, Any],
        ) -> List[Dict[Text, Any]]:
        # Your Adafruit IO credentials

        adafruit_io_username = "TuanNT"
        adafruit_io_key = "aio_CdQT11M9w4MIlEV9RVY7Krnfa45H"

        # Get the intent and extract relevant information from the tracker
        intent = tracker.latest_message['intent'].get('name')

        # Adjust the following logic based on your specific Rasa intents
        if intent == 'turn_on_light':
            # Make a request to turn on the button on Adafruit server
            url = f"https://io.adafruit.com/api/v2/{adafruit_io_username}/feeds/button1/data"
            headers = {"X-AIO-Key": adafruit_io_key}
            data = {"value": "1"}
            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200:
                dispatcher.utter_message("Light turned on.")
            else:
                dispatcher.utter_message("Failed to turn on the button.")

        elif intent == 'turn_off_light':
            # Make a request to turn off the button on Adafruit server
            url = f"https://io.adafruit.com/api/v2/{adafruit_io_username}/feeds/button1/data"
            headers = {"X-AIO-Key": adafruit_io_key}
            data = {"value": "0"}
            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200:
                dispatcher.utter_message("Light turned off.")
            else:
                dispatcher.utter_message("Failed to turn off the button.")

        return []


class AddEventToCalendar(Action):

    def name(self) -> Text:
        return "action_add_event"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        event_name = tracker.get_slot('event')
        time = tracker.get_slot('time')

        new_time = datetime.strptime(time, '%d/%m/%y %H:%M:%S') - timedelta(days=1) + timedelta(hours=10)

        add_event(event_name, new_time)

        dispatcher.utter_message(text="Event Added")

        return [AllSlotsReset()]


class getEvent(Action):

    def name(self) -> Text:
        return "action_get_event"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        event_name = get_event()

        print(event_name)
        # confirmed_event = tracker.get_slot(Any)
        dispatcher.utter_message(text="got events {name}".format(name=event_name))
        return []


# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/calendar']

CREDENTIALS_FILE = 'credentials.json'


def get_calendar_service():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('calendar', 'v3', credentials=creds)
    return service


def add_event(event_name, time):
    service = get_calendar_service()
    end = (time + timedelta(hours=1)).isoformat()

    event_result = service.events().insert(calendarId='a9caf84e59ea870a6b8c2db10dbb850593b86dfd11ce90c06073e8acc4ebbced@group.calendar.google.com',
                                           body={
                                               "summary": event_name,
                                               "description": 'Reminder',
                                               "start": {"dateTime": time.isoformat(), "timeZone": 'Etc/GMT+7'},
                                               "end": {"dateTime": end, "timeZone": 'Etc/GMT+7'},
                                           }
                                           ).execute()

    print("created event")
    print("id: ", event_result['id'])
    print("summary: ", event_result['summary'])
    print("starts at: ", event_result['start']['dateTime'])
    print("ends at: ", event_result['end']['dateTime'])


def get_event():
    service = get_calendar_service()
    now = datetime.utcnow().isoformat() + 'Z'
    events = service.events().list(calendarId='primary', timeMin=now,
                                   maxResults=10, singleEvents=True,
                                   orderBy='startTime').execute().get("items", [])

    print(events[0]["summary"])
    return events[0]["summary"]


def do_event():
    service = get_calendar_service()
    now = datetime.utcnow().isoformat() + 'Z'
    events = service.events().list(calendarId='a9caf84e59ea870a6b8c2db10dbb850593b86dfd11ce90c06073e8acc4ebbced@group.calendar.google.com', timeMin=now,
                                   maxResults=10, singleEvents=True,
                                   orderBy='startTime').execute().get("items", [])

    print(events[0]["end"])
    return events[0]["end"]




class ActionDoEvent(Action):

    def name(self) -> Text:
        return "action_do_event"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        event_name = do_event()

        print(event_name)
        dispatcher.utter_message(text="got events {name}".format(name=event_name))
        return []


class ActionDefaultFallback(Action):
    def name(self) -> Text:
        return "action_default_fallback"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # Get user message from Rasa tracker
        user_message = tracker.latest_message.get('text')
        print(user_message)

        # Set up OpenAI API request
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Authorization': 'Bearer -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',  # Replace 'your_api_key' with your actual OpenAI API key
            'Content-Type': 'application/json'
        }
        data = {
            'model': "gpt-3.5-turbo",
            'messages': [
                {'role': 'system', 'content': 'You are an AI assistant for the user. You help to solve user queries'},
                {'role': 'user', 'content': 'You: ' + user_message}
            ],
            'max_tokens': 100
        }

        # Make the request to the OpenAI API
        response = requests.post(url, headers=headers, json=data)

        # Check for a successful response
        if response.status_code == 200:
            chatgpt_response = response.json()
            # Extract the response message from OpenAI API
            message = chatgpt_response['choices'][0]['message']['content']
            dispatcher.utter_message(message)
        else:
            # Handle error
            dispatcher.utter_message("Sorry, I couldn't generate a response at the moment. Please try again later.")
            # Revert user message which led to fallback.
            return [UserUtteranceReverted()]

        # Return an empty list to indicate the action was executed successfully
        return []



# Define your Adafruit IO credentials
ADAFRUIT_IO_USERNAME = "TuanNT"
ADAFRUIT_IO_KEY = "aio_CdQT11M9w4MIlEV9RVY7Krnfa45H"


class ActionInformWeather(Action):
    def name(self):
        return "action_inform_weather"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        # Retrieve necessary parameters from the tracker
        while self.is_running:
            feed_names = [
                "temperature",
                "humidity",
            "wind-speed"
            ]

            # Initialize Adafruit IO client
            aio = Client(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY)

            for feed_name in feed_names:
                # Fetch data from Adafruit IO
                try:
                    data = aio.receive(feed_name)
                # Extract relevant information from 'data' and send it to the user
                    dispatcher.utter_message(f"The {feed_name} is: {data.value}")
                except Exception as e:
                    dispatcher.utter_message(f"Error fetching data from {feed_name}: {str(e)}")

            return []
