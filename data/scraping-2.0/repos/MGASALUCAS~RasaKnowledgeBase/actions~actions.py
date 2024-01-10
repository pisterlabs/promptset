# from rasa_sdk import Action
# import openai
# import os

# class ActionGetAnswer(Action):
#     def name(self) -> str:
#         return "action_get_answer"

#     def run(self, dispatcher, tracker, domain):
#         # Get the user's question from the last user message
#         user_question = tracker.latest_message.get('text')

#         # Load the prompt
#         with open('data/prompt.txt', 'r') as file:
#             prompt = file.read()

#         # Combine user question with the prompt
#         combined_prompt = f"{prompt}\nUser Question: {user_question}"

#         # Call the OpenAI API to get the GPT-3 response
#         openai.api_key = "YOUR_OPENAI_API_KEY"
#         response = openai.Completion.create(
#             engine="text-davinci-002",
#             prompt=combined_prompt,
#             max_tokens=1000,
#             stop=["\n"]
#         )

#         # Get the GPT-3 response
#         answer = response.choices[0].text.strip()

#         # Check if the response is relevant to the document
#         if "I don't know" in answer or "I don't understand" in answer:
#             dispatcher.utter_message(text="I'm sorry, I couldn't find the information in the documents.")
#         else:
#             dispatcher.utter_message(text=answer)

#         return []



from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import os


class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Welcome to the our digital assistance! Start by just saying Hello. "
                                      "Let's get started!"
                                      )

        return []



from rasa_sdk import Action
import openai
import os
import json


class ActionGetAnswer(Action):
    def name(self) -> str:
        return "action_get_answer"
    
    def __init__(self):
        # Load the API key from config.json
        with open('secrets/config.json', 'r') as config_file:
            config = json.load(config_file)
            self.openai_api_key = config.get("openai_api_key", "")


    def run(self, dispatcher, tracker, domain):
        # Get the user's question from the last user message
        user_question = tracker.latest_message.get('text')

        # Load the prompt
        with open('prompt.txt', 'r') as file:
            prompt = file.read()

        # Load the content of the documents
        documents_path = "external_documents/"
        document_contents = ""
        for filename in os.listdir(documents_path):
            with open(os.path.join(documents_path, filename), 'r') as file:
                document_contents += file.read()

        # Combine user question, prompt, and document contents
        combined_prompt = f"{prompt}\n\n{document_contents}\n\nUser Question: {user_question}"

        # Call the OpenAI API to get the GPT-3 response
        openai.api_key = self.openai_api_key
        response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=combined_prompt,
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
            )

        # Get the GPT-3 response
        answer = response.choices[0].text.strip()

        # Check if the response is relevant to the document
        if "I don't know" in answer or "I don't understand" in answer:
            dispatcher.utter_message(text="I'm sorry, I couldn't find the information in the documents.")
        else:
            dispatcher.utter_message(text=answer)

        return []
