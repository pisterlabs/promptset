import csv
import requests
from html.parser import HTMLParser

import openai

rows = []
APIKEY = "Enter API KEY here"

def fileReader(path):
    file = open(path)

    type(file)

    csvreader = csv.reader(file)
    headers = []
    headers = next(csvreader)
    #header now contains all of the headers

    
    for row in csvreader:
        rows.append(row)
    #each row corresponds with 1 person's name

    file.close()

    idDict = dict()

    #id as key, everything in an array afterwards
    for row in rows:
        idDict[row[0]] = row[1:]

    #header as key, all values under header in array
    headerDict = dict()
    counter = 0
    for header in headers:
        tempList = []
        for row in rows:
            tempList.append(row[counter])
        headerDict[header] = tempList
        counter += 1

def gptsomething():
    response = openai.OpenAI(api_key='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').chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Could you help analyze these purchase and give me some recommendations?"},
            {"role": "user", "content": "I love cookies!"}
        ]
    )
    
    return response.choices[0].message.content

def ctrlsomething(row1, row2):
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {APIKEY}"}
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    # data = query("Can you please let us know more details about your house")
    # data = query("Am I The Asshole for punishing my 16 year old stepdaughter after we found she was bullying a kid for being poor?")
    # data = query({"inputs": "Would you pick a house with a 3000 square feet with 2 bedrooms and 3 bathrooms, with a 1 hour commute time in a safe area" + \
    #              ", or a house with 5000 square feet with 1 bedrooms and 1 bathrooms, with a 5 minute commute time but in a somewhat risky area? I prefer the [MASK]", "options": {"wait_for_model": "true"}})
    data = query("Can you help me pick between the two houses with these stats? House 1: " + str(row1) + "House 2: " + str(row2))
    return data

for i in range(len(rows) - 1):
    print(ctrlsomething(row[i], row[i + 1]))
