import os
from typing import Any, Text, Dict, List
import pandas as pd
import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import openai 
import json

class RestaurantAPI(object):

    def __init__(self):
        self.db = pd.read_csv("restaurants.csv")

    def fetch_restaurants(self):
        return self.db.head()

    def format_restaurants(self, df, header=True) -> Text:
        return df.to_csv(index=False, header=header)


class ChatGPT(object):

    def __init__(self):
        self.url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-3.5-turbo"
        self.headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }
        self.prompt = "Answer the following question, based on the data shown. " \
            "Answer in a complete sentence and don't say anything else."

    def ask(self, restaurants, question):
        content  = self.prompt + "\n\n" + restaurants + "\n\n" + question
        body = {
            "model":self.model, 
            "messages":[{"role": "user", "content": content}]
        }
        result = requests.post(
            url=self.url,
            headers=self.headers,
            json=body,
        )
        return result.json()["choices"][0]["message"]["content"]
    

def ask_distance(restaurant_list):
    content = "measure the least distance with each given restaurant" +'/n/n' + restaurant_list
    completion = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=[{"role": "user", "content": content}],
    functions=[
    {
        "name": "get_measure",
        "description": "Get the least distance",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "list of all the restaurants and distances as a dictionary(restuarant_name:distance)",
                },
            },
            "required": ["distance"],
        },
    }
        ],
        function_call={"name":"get_measure"}
    )
    return completion.choices[0].message



restaurant_api = RestaurantAPI()
chatGPT = ChatGPT()

class ActionShowRestaurants(Action):

    def name(self) -> Text:
        return "action_show_restaurants"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        restaurants = restaurant_api.fetch_restaurants()
        results = restaurant_api.format_restaurants(restaurants)
        readable = restaurant_api.format_restaurants(restaurants[['Restaurants', 'Rating']], header=False)
        dispatcher.utter_message(text=f"Here are some restaurants:\n\n{readable}")

        return [SlotSet("results", results)]


def get_distance(d):
    d = json.loads(d)
    for i in d.keys():
        d[i]= float(d[i])
    t = min(d, key =d.get)
    return t

class ActionRestaurantsDetail(Action):
    def name(self) -> Text:
        return "action_restaurants_detail"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        previous_results = tracker.get_slot("results")
        question = tracker.latest_message["text"]
        answer = chatGPT.ask(previous_results, question)
        dispatcher.utter_message(text = answer)


class ActionRestaurantsDistance(Action):
    def name(self) -> Text:
        return "action_distance"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        previous_results = tracker.get_slot("results")
        func_calling= ask_distance(previous_results)
        reply_content = func_calling.to_dict()['function_call']['arguments']
        distance = json.load(reply_content)['distance']
        dispatcher.utter_message(text = get_distance(distance))