import json
import openai
from datetime import datetime
import re

use_gpt_request = False

def send_request(request_message):
    with open('api_key.txt', 'r') as f:
        api_key = f.read().strip()
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=request_message
    )
    print(response)
    return response["choices"][0]["message"]["content"]

# Get today's date
today = datetime.now().strftime('%m-%d')

# Create the prompt with the current date
# Create the prompt with the current date
prompt = f"Today is {today}. Please suggest 3-4 unique, creative, and obscure personal records I can attempt today that are somehow related to or inspired by this date. Remember, the personal records should be something I can do at one time, should not require any specialty equipment, and should be really original and interesting, not normal records though. They can have to do with obscure meditative practices, endurance, speed of accomplishment, anything obscure and unusual. Please provide your suggestions in a simple dictionary format with a name and a description, like this: {{'Name': 'Record Name', 'Description': 'Record Description'}}. Separate each suggestion with the delimiter '|||'. Do not include any follow-up explanations. Use double quotes in the response."

with open("PR_suggestions.txt", "r") as f:
    response_str = f.read()
#response_str.replace('"',"'")


if use_gpt_request:
    # Send the request to ChatGPT
    request_message = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    response_str = send_request(request_message)

print(response_str)

response_str = response_str.strip()

suggestion_pattern = re.compile(r"\{.*?\}", re.DOTALL)
suggestions_str = re.findall(suggestion_pattern, response_str)
suggestions = []

for s in suggestions_str:
    name_pattern = re.compile(r"'Name': '(.*?)'")
    name = re.search(name_pattern, s).group(1)

    description_pattern = re.compile(r"'Description': '(.*?)'")
    description = re.search(description_pattern, s).group(1)

    suggestions.append({"Name": name, "Description": description})

with open('/home/lunkwill/projects/tail/obsidian_dir.txt', 'r') as f:
    obsidian_dir = f.read().strip()

personal_records_file = obsidian_dir+'personal_records.txt'

# Load the existing personal records
with open(personal_records_file, "r") as f:
    personal_records = json.load(f)

# Iterate through the suggestions and ask the user if they like each one
for suggestion in suggestions:
    print(f"Suggestion: {suggestion['Name']} - {suggestion['Description']}")
    user_input = input("Do you like this suggestion? (yes/no): ").lower()

    # If the user likes the suggestion, add it to the personal_records dictionary
    if user_input == "y":
        personal_records[suggestion["Name"]] = {
            "records": {},
            "description": suggestion["Description"],
        }

# Save the updated personal_records to the file
with open(personal_records_file, "w") as f:
    json.dump(personal_records, f, indent=2)

print("Updated personal_records.txt with your selections.")
