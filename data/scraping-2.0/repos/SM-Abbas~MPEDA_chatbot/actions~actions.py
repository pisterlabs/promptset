# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
import csv
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests
import openai

def read_csv_data(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append(row)
    return data

class ActionFetchData(Action):
    def name(self) -> str:
        return "action_fetch_data"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        # Replace YOUR_API_KEY with your actual API key
        openai.api_key = '##################################'  # Replace with your OpenAI API key

        # Read the CSV data
        csv_data = read_csv_data("data/ExportersDirectory.csv")

        # Fetch the data from the URLs in the CSV data
        responses = []
        for row in csv_data:
            url = row[0]
            content = self.fetch_content(url)
            if content:
                responses.append(f"Data from {url}: {content}")

        # Respond to the user with the data
        if responses:
            response = "\n".join(responses)
        else:
            response = "Sorry, I couldn't fetch data from any of the URLs."

        dispatcher.utter_message(text=response)
        return []

    @staticmethod
    def fetch_content(url):
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return None
