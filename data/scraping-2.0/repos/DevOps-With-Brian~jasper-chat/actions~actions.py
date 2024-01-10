# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions
import requests
import os
from ghapi.all import GhApi
from dotenv import load_dotenv
import json
from datetime import datetime
import openai

load_dotenv()

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionAPOD(Action):

    def name(self) -> Text:
        return "action_apod"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # NASA APOD API endpoint
        api_url = "https://api.nasa.gov/planetary/apod"
        public_apod_url = "https://apod.nasa.gov/apod/"


        # API key
        nasa_api_key = os.getenv("NASA_API_KEY")

        # Make a GET request to the endpoint with the API key
        response = requests.get(api_url + "?api_key=" + nasa_api_key)

        # Check the status code of the response
        if response.status_code == 200:
            # If the status code is 200 (OK), get the JSON data from the response
            data = response.json()

            # Print the title and explanation of the APOD
            jasper_response = "Sure, here is a link to the Astronomy Picture of the Day, today's picture is of: {0} and you can see it at {1}".format(data["title"],public_apod_url)
            print("Title:", data["title"])
            print("Link to Picture & Full Explanation:", public_apod_url)
            dispatcher.utter_message(text=jasper_response)

        else:
            # If the status code is not 200 (OK), print an error message
            jasper_err = "Sorry, I wasn't able to hit the NASA API at this time."
            print("An error occurred while fetching the data:", response.text)
            dispatcher.utter_message(text=jasper_err)

        return []


class ActionSnaketemp(Action):

    def name(self) -> Text:
        return "action_snake_temp"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Fauna GraphQL endpoint
        now = datetime.now() # current date and time
        accessToken = os.getenv('FAUNA_KEY')
        endpoint = f"https://graphql.us.fauna.com/graphql"
        headers = {
            'Authorization': f"Bearer {accessToken}"
        }

        date = now.strftime("%Y-%m-%d")

        query = """query {
        tempsByDate(date: "%s"){
            data{
                _id
                temp
                time
                date
            }
        }
        }""" % (date)

        json_data = {
            'query': query
        }

        response = requests.post(endpoint, headers=headers, json=json_data)

        # Check the status code of the response
        if response.status_code == 200:
            # If the status code is 200 (OK), get the JSON data from the response
            print(response.json())
            temp_results = response.json()['data']
            temp_list = temp_results['tempsByDate']['data']
            latest_temp_index = (len(temp_list) - 1)
            print(latest_temp_index)
            latest_temp = temp_list[latest_temp_index]['temp']
            latest_temp_time = temp_list[latest_temp_index]['time']

            # Print the title and explanation of the APOD
            jasper_response = "The latest temperature reading from the snake room is currently {0}F and was taken at {1}".format(latest_temp, latest_temp_time)
            dispatcher.utter_message(text=jasper_response)

        else:
            # If the status code is not 200 (OK), print an error message
            jasper_err = "Sorry, I wasn't able to hit the graphql endpoint at this time."
            print("An error occurred while fetching the data:", response.text)
            dispatcher.utter_message(text=jasper_err)

        return []


class ActionJasperGHCount(Action):

    def name(self) -> Text:
        return "action_jasper_gh_action_count"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # GitHub API endpoint
        github_token = os.getenv("GITHUB_TOKEN")
        api = GhApi(owner='DevOps-With-Brian', repo='jasper-chat', token=github_token)
        jasper_actions = api.actions.list_workflow_runs_for_repo(ref='heads/main')
        jasper_build_count = jasper_actions['total_count']

        # Print the total count of builds
        jasper_response = "Sure, the current total number of builds that have ran for my code is currently {}".format(jasper_build_count)
        dispatcher.utter_message(text=jasper_response)

        return []
    

class ActionGithubPR(Action):

    def name(self) -> Text:
        return "action_github_pr"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # GitHub API endpoint
        github_token = os.getenv("GITHUB_TOKEN")
        api = GhApi(owner='DevOps-With-Brian', repo='jasper-chat', token=github_token)
        jasper_pull_requests = api.pulls.list(state='open')

        total_count_prs = len(jasper_pull_requests)

        if total_count_prs:
            jasper_response = "Yes I currently have {} pull requests open, for more information about them please see https://github.com/DevOps-With-Brian/jasper-chat/pulls".format(total_count_prs)
            dispatcher.utter_message(text=jasper_response)
        else:
            jasper_response = "No I currently don't have any open pull requests, yay!"
            dispatcher.utter_message(text=jasper_response)

        return []
    

class ActionChuckNorris(Action):

    def name(self) -> Text:
        return "action_chuck_norris"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # chuck norris API endpoint
        api_url = "https://api.chucknorris.io/jokes/random?category=dev"

        response = requests.get(api_url)

        # Check the status code of the response
        if response.status_code == 200:
            # If the status code is 200 (OK), get the JSON data from the response
            json_response = response.json()

            print(json_response)

            joke = json_response['value']

            # Print the title and explanation of the APOD
            jasper_response = joke
            dispatcher.utter_message(text=jasper_response)

        else:
            # If the status code is not 200 (OK), print an error message
            jasper_err = "Sorry, Chuck Norris appears to be taking a break atm...."
            print("An error occurred trying to get a joke from chuck", response.content)
            dispatcher.utter_message(text=jasper_err)

        return []
    

class ActionOOS(Action):

    def name(self) -> Text:
        return "action_oos"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        print(tracker.latest_message)

        user_message = tracker.latest_message.get("text")
        openai.api_key = os.getenv("OPENAI_KEY")

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=user_message,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )

        reply = response.choices[0].text.strip()

        dispatcher.utter_message(text=reply)

        return []