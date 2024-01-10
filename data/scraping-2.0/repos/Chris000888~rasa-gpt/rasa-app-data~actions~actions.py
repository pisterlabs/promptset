import openai
from rasa_sdk import Action, Tracker 
from typing import Any, Text, Dict, List
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import FollowupAction

def gpt_response(question):
    # OpenAI API Key
    openai.api_key = "YOUR_API_KEY"

    # Use OpenAI API to get the response for the given user text and intent
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt= question,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    ).choices[0].text

    # Return the response from OpenAI
    return response

class GPTAction(Action):

    def name(self) -> Text:
        return "action_default"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        user_sentence = tracker.latest_message['text']
        intent = tracker.latest_message.get('intent').get('name')
        entities = tracker.latest_message.get('entities')
        dispatcher.utter_message(gpt_response(user_sentence))
        return []
