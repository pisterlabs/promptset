import requests
import json
import openai
import os

API_KEY_AIDEVS = os.environ.get("API_KEY_AIDEVS")  # my env variable
API_KEY_OPENAI = os.environ.get("API_KEY_OPENAI")  # my env variable

openai.api_key = API_KEY_OPENAI


# token download function from AI Devs
def get_auth_token(task_name, print_json=False):
    url = "https://zadania.aidevs.pl/token/" + task_name
    apikey = API_KEY_AIDEVS

    # creating dictionary
    data = {
        "apikey": apikey
    }
    headers = {
            "Content-Type": "application/json"
    }

    # Serialize the dictionary to JSON
    data_json = json.dumps(data)
    # sending post request
    response = requests.post(url, data=data_json, headers=headers)

    if response.status_code == 200:
        response_json = response.json()
        if print_json:
            print(response_json)
        return response_json['token']
    else:
        print(f"Failed to get response:(. Error code: {response.status_code}")


# function to retrieve task content from AI Devs
def get_task(token_key, print_task=False):
    url = f"https://zadania.aidevs.pl/task/{token_key}"
    response = requests.get(url)
    response_json = response.json()

    # print information about task
    if print_task:
        print('----------- Task description -----------')
        for key in list(response_json.keys())[1:]:
            print(f'{key}: {response_json[key]}')
        print('-----------    ----------    -----------')

    return response_json


# function to send responses to AI Devs
def send_answer(token, answer_value, print_answer=False):
    url = "https://zadania.aidevs.pl/answer/" + token

    # Create a dictionary with the 'answer' field as an array
    data = {
        "answer": answer_value
    }
    headers = {
        "Content-Type": "application/json"
    }

    # Serialize the dictionary to JSON
    data_json = json.dumps(data)
    # sending post request
    response = requests.post(url, data=data_json, headers=headers)

    if print_answer:
        print(f"answer: {data}")

    if response.status_code == 200:
        print('Sending answer: done! :)')
    else:
        print(f"Failed to get response:(. Error code: {response.status_code}")
        print(f"Reason: {response.reason}")
        print(f"Text: {response.text}")

