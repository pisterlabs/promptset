# By Rohan "RJ" Jaiswal, using the OSU API
from urllib.request import urlopen
from dotenv import load_dotenv 
import ast
import json
import openai
import os

# generates a list of prerequisites from the course description
# TODO: add batch processing and concurrent requests to speed up
def parse_prerequisites(desc, prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": desc
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    generated_text = response['choices'][0]['message']['content']
    clean_text = generated_text.replace('\\"', '"')
    prerequisites = ast.literal_eval(clean_text)
    return prerequisites

def assemble(data, course, prompt):
    # parse_prerequisites works weirdly, everything else is fine
    _, prerequisites = course["description"].split(".", maxsplit=1)
    data[course["subject"] + "-" + course["catalogNumber"]] = {
        "classNumber": course["catalogNumber"],
        "department": course["subject"],
        "name": course["title"],
        "semester": course["term"],
        "credits": course["minUnits"],
        "description": course["description"],
        # "dependencies": parse_prerequisites(prerequisites, prompt),
    }


if __name__ == "__main__":
    import requests
    load_dotenv('.env.local')
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Define the base URL
    base_url = "https://content.osu.edu/v2/classes/search"

    # Define the query parameters
    params = {
        "q": "CSE",
        # "subject": "cse",
        # "term": "1238" # 23-au
    }

    # Send the GET request with the query parameters
    response = requests.get(base_url, params=params)

    # Parse the JSON response
    cse_json = response.json()

    with open("promptDirect.txt") as file:
        my_prompt = file.read()

    # empty dict to store data from
    data = {}
    for c in cse_json["data"]["courses"]:
        if c["course"]["catalogNumber"] not in data.keys():
            assemble(data=data, course=c["course"], prompt=my_prompt)


    json_data = json.dumps(data)

    # Writing to cse.json
    with open("cse2.json", "w") as outfile:
        outfile.write(json_data)
