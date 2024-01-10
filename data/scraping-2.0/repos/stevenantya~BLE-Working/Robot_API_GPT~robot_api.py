import requests
import json
import os
import openai
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_location():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "There is 6 location: A,B,C,D,E,Table. I want you to determine what location should I go to. Respond by giving the letter only e.g. A"},
            {"role": "user", "content": "I am having a meeting in location B. Let us go there now!"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "The team is holding a party in location C"}
        ]
    )
    return response['choices'][0]['message']['content']

goal_url = "http://192.168.201.100/api/NaviBee/robottask"

# Define the locations
locations = ['A', 'B', 'C', 'D', 'E', 'Table']

data = {}  # Dictionary to hold the content of the JSON files for each location

# Loop over each location to open and read the JSON files
for location in locations:
    with open(f'location_{location}.json', 'r') as file:
        data[location] = json.load(file)

next_location = get_location()
response = requests.post(goal_url, json= data[next_location])

post_response_json = response.json()
print(next_location)
print(response.status_code)
print(post_response_json)