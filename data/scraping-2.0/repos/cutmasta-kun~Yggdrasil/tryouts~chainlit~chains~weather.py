from typing import TypedDict, Union
import requests
from geopy.geocoders import Nominatim
from litechain.contrib import OpenAIChatChain, OpenAIChatMessage, OpenAIChatDelta
import json

class WeatherReturn(TypedDict):
    location: str
    forecast: str
    temperature: str

def get_current_weather(location: str) -> OpenAIChatDelta:
    """
    Gets the current weather in a given location, use this function for any questions related to the weather

    Parameters
    ----------
    location
        The city to get the weather, e.g. San Francisco. Guess the location from user messages
    """

    # Get the geographical coordinates of the location
    geolocator = Nominatim(user_agent="myGeocoder")
    location_obj = geolocator.geocode(location)
    print(f"\n\n{location_obj}\n\n")
    lat, lon = location_obj.latitude, location_obj.longitude

    # Define the URL for the GET request
    url = f"https://api.brightsky.dev/current_weather?lat={lat}&lon={lon}"

    # Make the GET request
    response = requests.get(url)
    print(f"\n\n{response}\n\n")

    # Parse the response
    weather_data = response.json()["weather"]

    result = {
        "location": location,
        "forecast": weather_data["condition"],
        "temperature": f"{weather_data['temperature']} C",
    }

    return OpenAIChatDelta(
        role="function", name="get_current_weather", content=json.dumps(result)
    )

weather_chain = OpenAIChatChain[str, OpenAIChatDelta](
    "WeatherChain",
    lambda user_input: [
        OpenAIChatMessage(role="user", content=user_input),
    ],
    model="gpt-3.5-turbo",
    functions=[get_current_weather],
    temperature=0,
)
