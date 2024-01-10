# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import UserUtteranceReverted
from rasa_sdk.events import SlotSet
from rasa_sdk.events import FollowupAction

import os
import openai


class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hello World!")

        return []
    

class ActionGptJoke(Action):
    """
    Custom action for providing jokes using the OpenAI GPT-3.5 Turbo model. It communicates with the model
    to generate a new joke based on specific rules. The rules include requesting a single joke in a specified
    format and avoiding repetition of jokes provided before.
    """

    def name(self) -> Text:
        self.openai=openai
        self.openai.api_key = os.getenv("OPENAI_API_KEY")
        self.previos_jokes = []
        return "action_gpt_joke"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # last_message = tracker.latest_message['text']
        # print('in action_gpt_joke')
        content="You are a joke provider. Provide one joke to make people happy. Rule: \
            1. Only give me one joke. \
            2. The format of joke should be: \
                Joke question. \
                Joke answer. \
                For example: \
                    What do you call a pig that does karate? \
                    A pork chop. \
            3. Do not provide the joke which is provided before, here is a list of joke provided before(if the list is empty, it means that there is no joke provided before):"
        previos_jokes = ';'.join(self.previos_jokes)
        content += previos_jokes
        # print('content: ', content)

        completion = self.openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": content},
            ],
            temperature=1.5,
        )

        answer = completion.choices[0].message["content"]
        self.previos_jokes.append(answer)

        dispatcher.utter_message(text=answer)

        return []
    
class ActionSetEmotion(Action):
    """
    Custom action for setting the 'emotion' slot based on detected entities in the latest user message.
    """

    def name(self) -> Text:
        return "action_set_emotion"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        try:
            emotion_value = tracker.latest_message['entities'][0]['value'] if tracker.latest_message['entities'][0]['entity'] == 'emotion' else None
            print('emotion: ', emotion_value)
            return [SlotSet("emotion", emotion_value)]
        except:
            print('no emotion detected')
            return []
        
    
class ActionSetNoEmotion(Action):

    def name(self) -> Text:
        return "action_set_no_emotion"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        return [
            SlotSet("emotion", None)
        ]

class ActionJokeResponse(Action):
    """
    Custom action for providing differnt response based on the detected emotion.
    """

    def name(self) -> Text:
        return "action_joke_response"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        emotion_value = tracker.get_slot('emotion')
        print('emotion: ', emotion_value)

        if emotion_value == "happy":
            dispatcher.utter_message(template="utter_response_happy_joke")
            return []
        elif emotion_value ==  "neutral" :
            dispatcher.utter_message(template="utter_response_neutral_joke")
            return []
        elif emotion_value == "sad":
            dispatcher.utter_message(template="utter_new_joke")
            dispatcher.utter_message(template="utter_think")
            return []
    
class ActionDefaultFallback(Action):
    """
    Executes the fallback action and goes back to the previous state of the dialogue.
    """

    def name(self) -> Text:
        return "action_default_fallback"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(template="utter_please_rephrase")

        # Revert user message which led to fallback.
        return [UserUtteranceReverted()]
