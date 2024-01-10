import requests
import json
from langchain.agents import load_tools
from serpapi import GoogleSearch

def post_pschat_message(incoming_msg):
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJVc2VySW5mbyI6eyJpZCI6MjE0MTgsInJvbGVzIjpbImRlZmF1bHQiXSwicGF0aWQiOiI1ZjE5ZmQzNS0yOGQyLTRhNjYtYmU5OC1mNjdhZGZiMmUyZTQifSwiaWF0IjoxNjg5MTAzMjY4LCJleHAiOjE2OTk0NzEyNjh9.EFDX15p4b8uxKhJcvwbMo6QUEpWvKKZw8-rOIcgmCUg",
        "Content-Type": "application/json"
        } 
    url = "https://api.psnext.info/api/chat"
    data = {
        "message": incoming_msg,
        "options": {
            "model": "gpt35turbo"
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    data = response.json()
    message_body = data['data']['messages'][2]['content']
    return message_body

def weather_message(latitude, longitude):
    weather_url = f'https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&units=imperial&appid=58cbf43603328cda648836bbdd9dd87c'
    weatherResponse = requests.get(weather_url)
    weatherData = weatherResponse.json()
    temp = weatherData['main']['temp']
    city = weatherData['name']
    feels_like = weatherData['main']['feels_like']
    message_body = f'Looks like you are in {city} where the current temperature is {round(temp)}°F but it feels like {round(feels_like)}°F.'
    return message_body

def getplaces_serp_api(incoming_msg):

    params = {
    "api_key": "7e3b9cc83717ae3f1fdc3a9d91b9a30227580c5d35a1f0164561755ae4fa880a",
    "engine": "google",
    "q": incoming_msg,
    "location": "Boston, Massachusetts, United States",
    "google_domain": "google.com",
    "gl": "us",
    "hl": "en"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    result_places = results['local_results']['places']
    places = []
    places_list = []  

    for place in result_places:
        unit = {}
        title = place['title']
        places.append(title)

        unit['title'] = place['title']
        unit['coordinates'] = place['gps_coordinates']
        places_list.append(unit)

    clean_response = ", ".join(places)
    raw_response = json.dumps({"places": places_list})

    return raw_response, clean_response
    
def getplaces_google_local_api(prompt, location):
    params = {
        "engine": "google_local",
        "q": prompt,
        "location": location,
        "api_key": "7e9c8419a21607c346150dcd3181a005e1f0e22aaf81be0f749feb0b34e0fb9e"
        }
    search = GoogleSearch(params)
    results = search.get_dict()
    local_results = results["local_results"]
    places = [] 
    places_info = []

    for place in local_results:
        if 'rating' in place:
            if place['rating'] > 4.2:
                unit = {}
                title = place['title']
                places.append(title)
                # getting street address
                address = place['address'].split(' · ')[1]
                place['gps_coordinates']['street_address'] = address
                # dictionary for each place
                unit["title"] = place['title']
                unit["coordinates"] = place['gps_coordinates']
                places_info.append(unit)
    return places, places_info