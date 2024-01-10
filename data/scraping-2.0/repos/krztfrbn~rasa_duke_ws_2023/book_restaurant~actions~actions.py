# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

import sqlite3

import dateparser
import math

import os
import openai

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()

# I'm assuming the following for my restaurant:
# 6 tables, each table for up to 4 guests
nr_tables = 6
max_nr_guests_per_table = 4

# Parties <= 4 get their own table
# Parties > 4 get the number of tables they require
# For each reservation we store date+time as string, number of guests as integer,
# and the name that the reservation is under (also string)

def get_nr_tables_needed(number_guests):
  nr_tables_needed = math.ceil(number_guests/max_nr_guests_per_table)
  print("number_guests: {0}, nr_tables_needed: {1}".format(number_guests, nr_tables_needed))
  return nr_tables_needed

def get_greeting_from_chatgpt(user_text):
  system_content = """You work in a restaurant called "Maeven's Bagel". 
       Your job is to answer the phone. If the caller gives 
       their name, then use that name in your reply. Give yourself
       a random name as well. If the caller is insulting, ask them to be kind. """
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": system_content},
      {"role": "user", "content": user_text}
    ]
  )
  #print(completion)
  return completion.choices[0].message.content

def get_chitchat_from_chatgpt(user_text):
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": """You are chatting with someone on the phone."""},
      {"role": "user", "content": user_text}
    ]
  )
  return completion.choices[0].message.content


class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hello World!")

        return []

class classActionGetGreetingChatGPT(Action):

    def name(self) -> Text:
        return "action_get_greeting_chatgpt"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
    
        user_text = tracker.latest_message['text']
        print(user_text)
        greeting = get_greeting_from_chatgpt(user_text)
        dispatcher.utter_message(text=greeting)

        return []
    
class classActionChitchatChatGPT(Action):

    def name(self) -> Text:
        return "action_get_chitchat_chatgpt"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
    
        user_text = tracker.latest_message['text']
        chitchat = get_chitchat_from_chatgpt(user_text)
        dispatcher.utter_message(text=chitchat)

        return []

class ActionCheckAvailability(Action):

    def name(self) -> Text:
        return "action_check_availability"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # If date, time, or number_guests is missing, let's not check
        # Instead, return the slot that should be filled next
        if tracker.get_slot('date') is None:
            return [SlotSet("next_slot_to_fill", 'date')]
        if tracker.get_slot('time') is None:
            return [SlotSet("next_slot_to_fill", 'time')]
        if tracker.get_slot('number_guests') is None:
            return [SlotSet("next_slot_to_fill", 'number_guests')]

        utterance = get_utterance_from_chatgpt()
        dispatcher.utter_message(text=utterance)

        conn = sqlite3.connect('../sqlite/restaurant-20231024.db') 
        cursor = conn.cursor()

        # Get all reservations (note: this could be improved at some point)
        sql = "SELECT * FROM reservations"
        cursor.execute(sql)
        records = cursor.fetchall()

        # Loop through existing bookings and count how many tables are already reserved
        # A table counts as reserved if it is +/- 2 hours from the requested time
        booking_available = False
        date_time_request = dateparser.parse(tracker.get_slot('date') + ' ' + tracker.get_slot('time'))
        nr_reserved_tables = 0
        nr_tables_needed = get_nr_tables_needed(int(tracker.get_slot('number_guests')))

        for row in records:
            date_time_booking = dateparser.parse(row[0])
            date_diff = date_time_booking - date_time_request
            print(date_diff.total_seconds())
            if date_diff.total_seconds() > -7200 and date_diff.total_seconds() < 7200:
                nr_reserved_tables += 1
        print("{0} tables are taken around that time".format(nr_reserved_tables))
        if nr_tables - nr_reserved_tables >= nr_tables_needed:
            booking_available = True
            print("Enough seating is available!")
            booking_available = True
        else:
            print("Not enough tables available")
            dispatcher.utter_message(text="I'm sorry but there's no table available at that time. What other time or day would work for you?")

        # Close connection
        conn.close()

        return [SlotSet("booking_available", booking_available), SlotSet("next_slot_to_fill", 'none')]
    
class ActionBookAppointment(Action):

    def name(self) -> Text:
        return "action_book_appointment"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        nr_tables_needed = get_nr_tables_needed(int(tracker.get_slot('number_guests')))

        conn = sqlite3.connect('../sqlite/restaurant-20231024.db') 
        cursor = conn.cursor()

        date_time_string = tracker.get_slot('date') + ' ' + tracker.get_slot('time')
        date_time = dateparser.parse(date_time_string).strftime("%m/%d/%Y, %H:%M:%S")

        nr_guests_remaining = int(tracker.get_slot('number_guests'))
        for t in range(nr_tables_needed):
            nr_guests = 4 if nr_guests_remaining > 3 else nr_guests_remaining
            sql="INSERT INTO reservations (date, reserved_under, nr_guests) VALUES (?, ?, ?);"
            cursor.execute(sql,(date_time, tracker.get_slot('name'), nr_guests))
            # Commit your changes in the database
            conn.commit()
            nr_guests_remaining -= 4

        # Close connection
        conn.close()

        return []