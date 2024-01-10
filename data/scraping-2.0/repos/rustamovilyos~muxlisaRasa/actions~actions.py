# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions
import random
import re
from typing import Any, Text, Dict, List

import yaml
from rasa_sdk import Action, Tracker
from rasa_sdk.events import (
    UserUtteranceReverted,
    ConversationPaused,
)
from rasa_sdk.executor import CollectingDispatcher


class ActionCheckQuestions(Action):
    def name(self) -> Text:
        return "action_check_exist_questions"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message['text']
        questions_1 = set()
        with open('../data/nlu.yml', 'r') as questions_file:
            questions = yaml.safe_load(questions_file)
            for intent_data in questions.get("nlu", []):
                intent_examples = intent_data.get("examples", "")
                intent_examples = re.split(r'[\n]', intent_examples.replace("- ", "").lower())
                questions_1.update(intent_examples)

        if user_message.lower() in questions_1:
            dispatcher.utter_message(text="well done!")
        else:
            dispatcher.utter_message(text="Iltimos savolingizni boshqacharoq shaklda bersangiz!")

        return []



#
#

# import openai
#
# openai.api_key = ""


class ActionDefaultFallback(Action):
    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher, tracker, domain):
        # output a message saying that the conversation will now be
        # continued by a human.
        current_state = tracker.current_state()
        latest_message = current_state["latest_message"]["text"]
        # context = "As an operator of 'Beeline' company, you are talking to a 'beeline' client and you must give answers in Uzbek"
        # message = latest_message
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": context},
        #         {"role": "user", "content": message},
        #     ],
        # )

        # reply = response["choices"][0]["message"]["content"]
        # print(reply)
        # reply = response["message"]["content"]
        message = "Kechirasiz! sizni nima demoqchi ekanligingizni tushuna olmadim."
        dispatcher.utter_message(text=message)
        # pause tracker
        # undo last user interaction
        return [ConversationPaused(), UserUtteranceReverted()]
