from fastapi import FastAPI,Query
from fastapi.middleware.cors import CORSMiddleware
from interface import Message,Recommendation,GetFlights
from typing import List
from openAI import PROMPT, MODEL, CUSTOMIZE_PROMPT
from functions import functions


import openai
import dotenv
import os
import json
import requests
import re
import httpx
# import googlemaps

try:
    dotenv.load_dotenv(".env")
except:
    pass

openai.api_key = os.environ["OPENAI_API_KEY"]
rapid_api_key = os.environ["RAPID_API_KEY"]
# google_api_key = os.environ["GOOGLE_API_KEY"]

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#Healtcheck
@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}


@app.post("/chat", response_model=dict)
async def chat(messages: List[Message], state: dict = None):
    response = on_message(messages, state)
    if type(response) is tuple:
        bot_response,arguments = response
        return {
            "botResponse": {"content": bot_response, "role": "assistant"},
            "newState": True,
            "arguments": arguments,
        }
    elif type(response) is str:
        return {"botResponse": {"content": response, "role": "assistant"}, "newState": False,"arguments": {}}


def on_message(message_history: List[Message], state: dict = None):
    if state is None or "counter" not in state:
        state = {"counter": 0}
    else:
        state["counter"] += 1
    # Generate Response use Chat Completion
    response = openai.ChatCompletion.create(
        temperature=0.7,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": PROMPT},
            *map(dict, message_history),
        ],
        functions=functions,
        function_call="auto",
        model=MODEL,
    )

    if response["choices"][0]["finish_reason"] == "function_call":
        # generate second response if all the necessary details are fetched from user
        data = json.loads(
            response["choices"][0]["message"]["function_call"]["arguments"]
        )
        destination = data["destination"]
        departure = data["departure"]
        startDate = data["trip start date"]
        endDate = data["trip end date"]
        budget = data["trip budget"]
        
        print(destination,departure,startDate,endDate,budget)

        hotels = get_hotel_details(departure)
        second_response = openai.ChatCompletion.create(
            temperature=0.7,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": PROMPT},
                *map(dict, message_history),
            ],
            model=MODEL,
        )

        final_response = (
            second_response["choices"][0]["message"]["content"]
            + "\n \n"
            + "Suggested Hotels for you to Book : "
            + "\n \n"
            + "Name : "
            + "taj hotel"
            + "\n \n"
            + "Link : "
            + "https"
            + "\n \n"
            + "Rating : "
            + "five",
            {"departure":departure, "destination":destination, "start_date": startDate, "end_date": endDate,"budget": budget}
        )

        return final_response
    else:
        return response["choices"][0]["message"]["content"]


@app.post("/recommendations", response_model=dict)
async def recommendations(data: Recommendation):
        print("data",data)
        response = openai.ChatCompletion.create(
        temperature=0.7,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": CUSTOMIZE_PROMPT.format(data.destination, data.departure, data.start_date, data.end_date, data.budget)},
        ],
        model=MODEL,
    )
        gptResponse = response["choices"][0]["message"]["content"]
        print(gptResponse)
        json_match = re.search(r'{.*}', gptResponse)
        json_data = json_match.group(0)
        json_dict = json.loads(json_data)
        # print(json_dict)
        return json_dict
            
    
    

def get_hotel_details(city):
    url = "https://best-booking-com-hotel.p.rapidapi.com/booking/best-accommodation"
    querystring = {"cityName":city,"countryName":"India"}
    headers = {
        "X-RapidAPI-Key": rapid_api_key,
        "X-RapidAPI-Host": "best-booking-com-hotel.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    return response.json()

@app.post("/flightsToDestination")
async def flightsToDestination(body: GetFlights):
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"https://flights.booking.com/api/flights/",
            params={
                "type": "ROUNDTRIP",
                "adults": body.adults,
                "cabinClass": "ECONOMY",
                "children": body.children,
                "from": body.fromLocation.city + ".CITY",
                "to": body.toLocation.city + ".CITY",
                "fromCountry": body.fromLocation.country,
                "toCountry": body.toLocation.country,
                "fromLocationName": body.fromLocation.cityName,
                "toLocationName": body.toLocation.cityName,
                "depart": body.departDate,
                "return": body.returnDate,
                "sort": "BEST",
                "travelPurpose": "leisure",
                "aid": 304142,
                "label": "gen173nr-1FEghwYWNrYWdlcyiCAjjoB0gzWARoJ4gBAZgBMbgBB8gBDNgBAegBAfgBAogCAagCA7gCvrPYpAbAAgHSAiQ5ZDQ1MDI0Ny1jMzEyLTQ3YzUtYWI5My0zN2EyYTcwNjk3ZjHYAgXgAgE",
                "enableVI": 1
            },
            headers={
                "Accept": "*/*",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/114.0",
                "Accept-Language": "en-US,en;q=0.5",
                "X-Requested-From": "clientFetch",
                "X-Flights-Context-Name": "search_results",
                "X-Booking-Affiliate-Id": "304142",
                "X-Booking-Label": "gen173nr-1FEghwYWNrYWdlcyiCAjjoB0gzWARoJ4gBAZgBMbgBB8gBDNgBAegBAfgBAogCAagCA7gCvrPYpAbAAgHSAiQ5ZDQ1MDI0Ny1jMzEyLTQ3YzUtYWI5My0zN2EyYTcwNjk3ZjHYAgXgAgE"
            },
            timeout=None
        )

        offers = res.json()
        if "error" in offers:
            # no flight
            return []

        offers = offers["flightOffers"]
        difference = 10000000
        closestToBudget = 0
        for i, offer in enumerate(offers[1:]):
            price = sum([item["travellerPriceBreakdown"]["total"]["units"] for item in offer["travellerPrices"]])
            newDiff = abs(price - body.flightBudget)

            if newDiff < difference:
                difference = newDiff
                closestToBudget = i + 1

        offerToUse = offers[closestToBudget]

        return {"details": offerToUse["segments"], "price": sum(
            [item["travellerPriceBreakdown"]["total"]["units"] for item in offerToUse["travellerPrices"]])}



# Replace 'YOUR_API_KEY' with your actual Google Maps API key
# gmaps = googlemaps.Client(key=google_api_key)

# @app.get("/get_location/")
# async def get_location(latitude: float = Query(..., description="Latitude of the location"),
#                        longitude: float = Query(..., description="Longitude of the location")):
 
#         reverse_geocode = gmaps.reverse_geocode((latitude, longitude))
        
#         if reverse_geocode:
#             location = reverse_geocode[0]
#             formatted_address = location.get("formatted_address", "")
#         else:
#             formatted_address = "Address not found"
        
#         return {
#             "location": formatted_address,
#             "latitude": latitude,
#             "longitude": longitude
#         }
