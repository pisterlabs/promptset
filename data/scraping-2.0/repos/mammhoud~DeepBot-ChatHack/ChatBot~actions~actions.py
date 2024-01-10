# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Text, Optional


from .slack_api import SlackApp
from rasa_sdk import Action, Tracker
from rasa_sdk.types import DomainDict
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import (
    SlotSet,
    UserUtteranceReverted,
    ConversationPaused,
    EventType,
    BotUttered,FollowupAction,ActionExecuted,SessionStarted,
)

import cohere 

import pandas
import requests
import googlesearch
import mysql.connector
import dis
from email.mime import image
import logging
import json
from datetime import datetime

# import pycountry


# from slack_sdk import WebClient
# from slack_sdk.errors import SlackApiError

import json
import random
import time
import os

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Text, Optional


import sqlite3

# Default parameters for DatabseConnection class. Can be overriden in constructor.
# change this to the location of your SQLite file
path_to_db = "../db.sqlite3"
db = ["localhost", "chatbot", "root", "p@Ssword"]
def_db = db

# Define this list as the values for the `language` slot. Arguments of the `get_..._lang` functions should respect this order.
lang_list = ["English", "Arabic"]  # Same as slot values
arabic_prompt="الان اريد مساعدتك ك متحدث اصطناعى (chatbot) اريد منك ان تلعب دور خدمة العملاء و سيتم اعطائك فى كل مرة سؤال يجب ان تجاوبة فى نطاق {} و سوف تجاوب بناءا عن شات بوت يجاوب من بيانات تم اعطائها خصيصا للاجابة عن مشاكل العملاء او تحويلة الى احد ممثلى خدمة العملاء هذة البيانات التى اعطيها لك لا اريد ان تعطى العميل تفاصيل عنها سؤال العميل هو {}".format("topic","question")
english_prompt="Now, I want to assist you as an artificial speaker (chatbot). I want you to play the role of customer service. Each time, you will be given a question that you must answer within the scope of {}. You will respond based on a chatbot that answers from data specifically given to answer customer problems or to transfer to one of the customer service representatives. This data that I give you, I don't want you to give the customer details about it. The customer's question is {}.".format("topic","question")

# Constants that will be used many times in the code.
text_does_it_work = ["Does it work now?", "هل يعمل الآن؟"]

text_anything_else = [
    "Anything else I can help with?",
    "أي شيء آخر يمكنني المساعدة به؟",
]

buttons_yes_no_emoji = [
    {'title': '👍', 'payload': '/affirm'},
    {'title': '👎', 'payload': '/deny'}]

button_stop_emoji = [{'title': '🚫', 'payload': '/stop'}]
buttons_yes_no_stop_emoji = buttons_yes_no_emoji + button_stop_emoji
  
        
####################################################################################################
#                                    DEFAULT RASA ACTIONS                                          #
####################################################################################################


class ActionSessionStart(Action):
    def name(self):
        return "action_session_start"

    @staticmethod
    def fetch_slots(tracker):
        slots = []
        slots_to_keep = []

        for slot_name in slots_to_keep:
            slot_value = tracker.get_slot(slot_name)
            if slot_value is not None:
                slots.append(SlotSet(key=slot_name, value=slot_value))

        return slots

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        print(tracker.sender_id)

        events = [SessionStarted()]
        events.extend(self.fetch_slots(tracker))
       # events.append(FollowupAction('action_virtual_ai'))
        events.append(ActionExecuted("action_listen"))

        return events

class SurveySubmit(Action):
    def name(self) -> Text:
        return "action_survey_submit"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(response="utter_open_feedback")
        dispatcher.utter_message(response="utter_survey_end")
        pandas.DataFrame.to_csv(tracker.latest_message,(username,phone_number,slot_values), "survey.csv")
        return [SlotSet("survey_complete", True)]
    
class GiveAge(Action):
    def name(self) -> Text:
        return "action_give_age"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        evt = BotUttered(text="my age is bot? idk", metadata={"ageGiven": "bot"})
        
        return [evt]


class GiveUid(Action):
    def name(self) -> Text:
        return "action_give_uid"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        evt = BotUttered(text="my uid is bot? idk", metadata={"uidGiven": "bot"})

        return [evt]


class GiveName(Action):
    def name(self) -> Text:
        return "action_give_name"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        evt = BotUttered(text="my name is bot? idk", metadata={"nameGiven": "bot"})

        return [evt]

####################################################################################################
#                                    Bot Shopping Actions                                          #
####################################################################################################


class ActionProductSearch(Action):
    def name(self) -> Text:
        return "action_product_search"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # connect to DB
        connection = sqlite3.connect(path_to_db)
        cursor = connection.cursor()

        # get slots and save as tuple
        product = [(tracker.get_slot("color")), (tracker.get_slot("size"))]

        # place cursor on correct row based on search criteria
        cursor.execute("SELECT * FROM products_inventory WHERE color=? AND size=?", product)

        # retrieve sqlite row
        data_row = cursor.fetchone()

        if data_row:
            # provide in stock message
            dispatcher.utter_message(response="utter_in_stock")
            connection.close()
            slots_to_reset = ["size", "color"]
            return [SlotSet(slot, None) for slot in slots_to_reset]
        else:
            # provide out of stock
            dispatcher.utter_message(response="utter_no_stock")
            connection.close()
            slots_to_reset = ["size", "color"]
            return [SlotSet(slot, None) for slot in slots_to_reset]



class OrderStatus(Action):
    def name(self) -> Text:
        return "action_order_status"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # connect to DB
        connection = sqlite3.connect(path_to_db)
        cursor = connection.cursor()

        # get email slot
        order_email = (tracker.get_slot("email"),)

        # retrieve row based on email
        cursor.execute("SELECT * FROM consumers_order WHERE order_email=?", order_email)
        data_row = cursor.fetchone()

        if data_row:
            # convert tuple to list
            data_list = list(data_row)

            # respond with order status
            dispatcher.utter_message(response="utter_order_status", status=data_list[5])
            connection.close()
            return []
        else:
            # db didn't have an entry with this email
            dispatcher.utter_message(response="utter_no_order")
            connection.close()
            return []


class CancelOrder(Action):
    def name(self) -> Text:
        return "action_cancel_order"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # connect to DB
        connection = sqlite3.connect(path_to_db)
        cursor = connection.cursor()

        # get email slot
        order_email = (tracker.get_slot("email"),)

        # retrieve row based on email
        cursor.execute("SELECT * FROM consumers_order WHERE status='order pending' and order_email=?", order_email)
        data_row = cursor.fetchone()

        if data_row:
            # change status of entry
            status = [("cancelled"), (tracker.get_slot("email"))]
            cursor.execute("UPDATE consumers_order SET status=? WHERE order_email=?", status)
            connection.commit()
            connection.close()
            # tracker.get_slot("email") , 
            # confirm cancellation
            
            dispatcher.utter_message(response="utter_order_cancel_finish")
            return []
        else:
            # db didn't have an entry with this email
            dispatcher.utter_message(response="utter_no_order")
            connection.close()
            return []


class ReturnOrder(Action):
    def name(self) -> Text:
        return "action_return"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # connect to DB
        connection = sqlite3.connect(path_to_db)
        cursor = connection.cursor()

        # get email slot
        order_email = (tracker.get_slot("email"),)

        # retrieve row based on email
        cursor.execute("SELECT * FROM consumers_order WHERE order_email=?", order_email)
        data_row = cursor.fetchone()

        if data_row:
            # change status of entry
            status = [("returning"), (tracker.get_slot("email"))]
            cursor.execute("UPDATE consumers_order SET status=? WHERE order_email=?", status)
            connection.commit()
            connection.close()

            # confirm return
            dispatcher.utter_message(response="utter_return_finish")
            return []
        else:
            # db didn't have an entry with this email
            dispatcher.utter_message(response="utter_no_order")
            connection.close()
            return []



####################################################################################################
#                                            HANDOFF                                               #
####################################################################################################

class ActionRequestHuman(Action):
    def name(self):
        return "action_request_human"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        text = ""

        if tracker.get_slot("username") and tracker.get_slot("login_type"):
            username = tracker.get_slot("username").title()
            login_type = tracker.get_slot("login_type")
            phone_number = username
            sender_id = tracker.sender_id
            slot_values = list_slots(tracker, list(domain["slots"].keys()))

            if login_type != "Phone_Number":
                try:
                    db = DatabaseConnection(db_info=def_db)
                    results = db.query(
                        "SELECT Username, Phone_Number "
                        "FROM `user_info` "
                        f"WHERE {login_type} = '{username}'"
                    )
                    username, phone_number = results[0]
                    db.disconnect()
                except Exception as e:
                    print(f"\n> ActionRequestHuman: [ERROR] {e}")
                    dispatcher.utter_message(
                        "Sorry, I couldn't connect to the database."
                    )
                    return []

            text = (
                get_text_from_lang(
                    tracker,
                    [
                        "You requested human help. Someone will contact you shortly on {}.".format(
                            phone_number
                        ),
                        "لقد طلبت مساعدة. سيتصل بك شخص ما قريبًا في {}.".format(
                            phone_number
                        ),
                    ],
                )
                + "\n"
                + get_text_from_lang(tracker, text_anything_else)
            )

            slack = SlackApp("batot")
            slack.sendMessage(
                f"{username} ({phone_number}) requested assistance.\nRasa Tracker sender ID: {sender_id}.\nSlots:\n{slot_values}"
            )

            print("\nBOT:", text)
            dispatcher.utter_message(text)
            pandas.DataFrame.to_csv((username,phone_number,slot_values), "survey.csv")


        else:
            text = get_text_from_lang(
                tracker,
                [
                    'You requested human help but are not logged in. Please type "log in" to log in.',
                    'تم ارسال طلبك و لكن لقد طلبت مساعدة لكنك لم تسجل الدخول. الرجاء كتابة "تسجيل الدخول" لتسجيل الدخول. عذرا لا يمكننا التواصل معك مباشرتا الان',
                ],
            )
            #dispatcher.utter_message(image="https://i.imgur.com/Z99thxA")

            ##print("\ntracker:", tracker.)
            print("\nBOT:", text)
            dispatcher.utter_message(text, image="https://i.imgur.com/Z99thxA")


        return []
############################################################################################
#                               API COHERE CALLS                                           #
############################################################################################
class ActionVirtualAI(Action):
    def name(self):
        return "action_virtual_ai"

    #chat_hist = []
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # dispatcher.utter_message(buttons = [
        #     {"payload": "/affirm", "title": "Yes"},
        #     {"payload": "/deny", "title": "No"},
        # ], text="Do you want to ask another questions?")        
        #user_message = {"user_name": "USER", "text": tracker.latest_message.get("text")}
        #bot_message = {"user_name": "CHATBOT", "text": response.text}
        #self.chat_hist.append(user_message)
        #dispatcher.utter_message(text="Do you want to ask another questions?")
        #ActionExecuted("action_listen")
        co_response = None
        if tracker.latest_message['intent'].get("name") in ("/affirm", "Yes","yes","continue","affirm"):
            dispatcher.utter_message(text="Do you want to ask a questions?❓")
            #ActionExecuted("action_listen")
            #dispatcher.utter_message(text="❓?")
        else:
            co_response = cohere.Client('#').chat(
            chat_history=[
                {"role": "CHATBOT", "message": tracker.latest_message.get("text")},
                {"role": "USER", "message": "❓"},
            ],
            message=tracker.latest_message.get("text"),  # Use the user's first message
            connectors=[{"id": "web-search"}],
            prompt_truncation="AUTO",
            documents=[],
            temperature=0.4,
        )
            dispatcher.utter_message(co_response.text) 

        # print(response.text)
        # Events to be sent back
        #dispatcher.utter_message(response="utter_optional_virtual_ai")
        dispatcher.utter_message(text="Do you want to ask another question ❓")
        FollowupAction("action_virtual_ai")
        # else:                
        #     return []
        
############################################################################################
#                                                                                          #
############################################################################################

# class ActionVirtualAI(Action):
#     def name(self):
#         return "action_virtual_ai"

#     def run(
#         self,
#         dispatcher: CollectingDispatcher,
#         tracker: Tracker,
#         domain: Dict[Text, Any],
#     ) -> List[Dict[Text, Any]]:
#         response = cohere.Client('#').chat(
#             chat_history=[
#                 {"role": "CHATBOT", "message": "Hey there, how can I help you?"},
#                 {"role": "USER", "message": "I thinking about that i have a question to AI virtual assistants."},
#             ],
#             message=tracker.latest_message.get("text"),  # Use the user's first message
#             connectors=[{"id": "web-search"}],  # You can add more connectors here
#         )
#         print(response.text)
#         # Events to be sent back
#         dispatcher.utter_message(response.text) 
#         dispatcher.utter_message(text="Do you want to ask another questions?")
#         # dispatcher.utter_message(buttons = [
#         #     {"payload": "/affirm", "title": "Yes"},
#         #     {"payload": "/deny", "title": "No"},
#         # ])        

#         ActionExecuted("action_listen")
#         if tracker.latest_message['intent'].get("name") == "/affirm":
#             FollowupAction("action_virtual_ai")
#         else:                
#             return []
############################################################################################
#                                 Stack Slots                                              #
############################################################################################

    
####################################################################################################
#                                           LANGUAGES                                              #                      
####################################################################################################

def get_lang(tracker):
    try:
        lang = tracker.slots['language'].title()
        return lang
    except Exception as e:
        return 'English'

def get_lang_index(tracker):
    return lang_list.index(get_lang(tracker))

''' utter_list is a list of outputs in multiple lanaguages, each output can be a string or a list of strings '''
def get_text_from_lang(tracker, utter_list = []):
    lang_index = get_lang_index(tracker)

    if not utter_list: # No text was given for any language
        return '[NO TEXT DEFINED]'

    if lang_index >= len(utter_list): # No text defined for current language
        lang_index = 0

    text = utter_list[lang_index]

    if isinstance(text, list): # If a list is given for the language, choose a random item
        text = str(text[random.randint(0,len(text)-1)])
    else:
        text = str(text)
    
    return text 
    
def get_response_from_lang(tracker, response):
    return response + '_' + get_lang(tracker)

def get_buttons_from_lang(tracker, titles = [], payloads = []):
    lang_index = get_lang_index(tracker)
    buttons    = []

    if lang_index >= len(payloads): # No text defined for current language
        lang_index = 0
    
    for i in range(min(len(titles[lang_index]), len(payloads))):
        buttons.append({'title': titles[lang_index][i], 'payload': payloads[i]})
    return buttons

class ActionUtterAskLanguage(Action):
    def name(self):
        return 'action_utter_ask_language'
    
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        
        text = get_text_from_lang(
            tracker,
            ['Choose a language:',
            ':اختر لغة'])

        buttons = [ # https://forum.rasa.com/t/slots-set-by-clicking-buttons/27629
            {'title': 'English',  'payload': '/set_language{"language": "English"}'},
            {'title': 'عربي',     'payload': '/set_language{"language": "Arabic"}'},
        ]
       
        print('\nBOT:', text, buttons)
        dispatcher.utter_message(text = text, buttons = buttons)
        return []

class ActionUtterSetLanguage(Action):
    def name(self) -> Text:
        return 'action_utter_set_language'
    
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        

        current_language = tracker.slots['language'].title()
        text = 'I only understand English, French, Arabic, and Armenian. The language is now English.'
        
        if current_language == 'English':
            text = 'The language is now English.'
        elif current_language == 'Arabic':
            text = 'اللغة الآن هي العربية.'

        print('\nBOT:', text)
        dispatcher.utter_message(text = text)
        
        if not tracker.get_slot('service_type'):
            return [FollowupAction('action_utter_service_types')]
        return []
