from fastapi import FastAPI
import requests
import json
import openai
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
import os
from datetime import datetime

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]

url = "https://test.api.amadeus.com/v1/security/oauth2/token"

payload = 'client_id=vAotiGVO4wbZALr8vL3slWD8LApwAuuu&client_secret=Zgx6CsnGE9QO2yd1&grant_type=client_credentials'
headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
}

response = requests.request("POST", url, headers=headers, data=payload)
auth_token = response.json()


openai.api_key = 'open-ai-key'

messages = [
    {"role": "system", "content": "You are a helpful Travel Guide."}
]

d = {}
s = ''


@app.post("/flights")
async def search_flights(returnDate="2023-05-08", originLocationCode="DEL", destinationLocationCode="GOI", departureDate="2023-05-04", adults="1"):
    if returnDate != None:
        url = "https://test.api.amadeus.com/v2/shopping/flight-offers?originLocationCode="+originLocationCode+"&destinationLocationCode=" + \
            destinationLocationCode+"&departureDate="+departureDate + \
            "&returnDate="+returnDate+"&adults="+adults+"&max=5&currencyCode=INR"
    else:
        url = "https://test.api.amadeus.com/v2/shopping/flight-offers?originLocationCode="+originLocationCode + \
            "&destinationLocationCode="+destinationLocationCode+"&departureDate=" + \
            departureDate+"&adults="+adults+"&max=5&currencyCode=INR"

    payload = {}
    headers = {
        'Authorization': 'Bearer '+auth_token["access_token"]
    }
    date1 = datetime.strptime(departureDate, '%Y-%m-%d')
    date2 = datetime.strptime(returnDate, '%Y-%m-%d')

    global s
    # calculate the difference between two dates and extract the number of days
    delta = date2 - date1
    num_days = delta.days
    s = f'Give {num_days} day itinerary to explore {destinationLocationCode} and come back to {originLocationCode}'
    response = requests.request("GET", url, headers=headers, data=payload)
    global d
    d = response.json()
    return response.json()


@app.post("/nearbyCities")
async def nearby_cities(cityCodes):
    url = "https://test.api.amadeus.com/v1/reference-data/recommended-locations?cityCodes="+cityCodes
    payload = {}
    headers = {
        'Authorization': 'Bearer '+auth_token["access_token"]
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    return response.text


@app.post("/chatgpt")
def update_chat(content, role="user"):
    messages.append({"role": role, "content": content})
    return summary(messages)


def summary(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    if response.choices[0].message != None:
        return response.choices[0].message["content"]
        # print(response.choices[0].message["content"])

    else:
        return 'Failed to Generate response!'


@app.get("/data")
async def get_data():
    print(d)
    dpt_name = []
    arr_name = []
    dpt_time = []
    arr_time = []
    clas = []
    price = []
    carrier = []
    for i in range(0, 5):
        dpt_name.append(d["data"][i]["itineraries"][0]
                        ["segments"][0]["departure"]["iataCode"])
        arr_name.append(d["data"][i]["itineraries"][0]
                        ["segments"][0]["arrival"]["iataCode"])
        dpt_time.append(d["data"][i]["itineraries"][0]
                        ["segments"][0]["departure"]["at"])
        arr_time.append(d["data"][i]["itineraries"][0]
                        ["segments"][0]["arrival"]["at"])
        price.append(d["data"][i]["price"]["total"])
        clas.append(d["data"][i]["travelerPricings"][0]
                    ["fareDetailsBySegment"][0]["cabin"])
        carrier.append(d["dictionaries"]["carriers"][d["data"]
                       [i]["itineraries"][0]["segments"][0]["carrierCode"]])
    return [dpt_name, arr_name, dpt_time, arr_time, clas, price, carrier]


@app.get("/prompt")
async def get_data():
    return s