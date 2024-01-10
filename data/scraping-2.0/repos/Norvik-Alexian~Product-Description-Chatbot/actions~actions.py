import os
import requests
import openai
import actions.config as config

from typing import Any, Text, Dict, List
from dotenv import load_dotenv
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_kEY')

if not OPENAI_API_KEY:
    raise EnvironmentError('You should set OPENAI_API_kEY as your environment variable.')


class ActionProductDescriptionGenerator(Action):
    openai.api_key = OPENAI_API_KEY

    def name(self) -> Text:
        return "action_product_description"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        model = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'user', 'content': f'{config.SHORT_DESCRIPTION_PROMPT}: jeans, blue, trendy, cotton'}
            ],
            temperature=0.8,
            max_tokens=100,
            top_p=1.,
            stop=None,
            n=1
        )
        generated_content = model.choices[0].message.content.strip()

        dispatcher.utter_message(text=generated_content)

        return []