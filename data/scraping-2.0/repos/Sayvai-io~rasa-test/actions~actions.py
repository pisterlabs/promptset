# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List

# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher


# class ActionHelloWorld(Action):

#     def name(self) -> Text:
#         return "action_hello_world"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

#         dispatcher.utter_message(text="Hello World!")

#         return []

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import openai

class OpenAILLMAction(Action):
    def name(self):
        return "action_generate_response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        user_message = tracker.latest_message.get("text")
        openai.api_key = "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"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=user_message,
            max_tokens=50  # Adjust as needed
        )
        generated_response = response.choices[0].text.strip()
        dispatcher.utter_message(text=generated_response)
        return []
