from app.functions import log, getConfig
from app.assistant.response_manager import openaiResponseCompletion

import datetime
import requests
import googlemaps
import json


def handle_weather(intent, entities, userMessage):
    gmaps = googlemaps.Client(key=getConfig()["otherKeys"]["googleMaps"])

    locationName = entities["weather_location"]["value"]
    locationGeocode = gmaps.geocode(locationName)[0]
    locationCoords = {
        "lat": locationGeocode["geometry"]["location"]["lat"],
        "lon": locationGeocode["geometry"]["location"]["lng"],
    }
    time = entities["weather_time"]["value"]

    log(locationCoords)

    if time == ("today" or "tomorrow"):
        weather = getWeatherHourly(locationCoords)
        messages = [
            {
                "role": "system",
                "content": "The user is asking about the weather. Give a concise report relating to the user's question based on the data provided.",
            },
            {"role": "system", "content": str(weather)},
            {"role": "user", "content": str(userMessage)},
        ]
        response = openaiResponseCompletion(messages)
        return {
            "overwrite": True,
            "message": response.choices[0].message.content,
        }
    if time == "this week":
        pass

    log(entities)

    return {
        "overwrite": True,
        "message": f"This feature is still under development.",
    }


def getWeatherHourly(location):
    apiKey = getConfig()["openWeatherMap"]

    resultsHourlyResult = requests.get(
        f"https://api.openweathermap.org/data/3.0/onecall?lat={location['lat']}&lon={location['lon']}&appid={apiKey}&units=imperial&exclude=minutely,alerts,daily"
    )

    resultsHourlyResult = resultsHourlyResult.json()
    resultsHourly = resultsHourlyResult
    resultsHourly["hourly"] = resultsHourly["hourly"][::3]

    keys_to_remove = [
        "pressure",
        "humidity",
        "dew_point",
        "clouds",
        "visibility",
        "pop",
    ]

    for hour in resultsHourly["hourly"]:
        for key in keys_to_remove:
            if key in hour:
                del hour[key]
        hour["time"] = datetime.datetime.fromtimestamp(hour["dt"]).strftime(
            "%d %B %Y %H:%M:%S"
        )

    return json.dumps(resultsHourly, separators=(",", ":"))
