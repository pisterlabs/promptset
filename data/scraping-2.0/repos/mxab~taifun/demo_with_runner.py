import os
import urllib.parse as urlparse

import httpx
import openai
import rich
from rich.prompt import Prompt
from pydantic import BaseModel, Field
from taifun import Taifun, TaifunConversationRunner

# This demo shows how to use the TaifunConversationRunner to run a Taifun
# We provide 3 functions:
# 1. get_location: asks the user for their location
# 2. get_lang_lat: gets the latitude and longitude of a location
# 3. get_current_weather: gets the current weather of a location


taifun = Taifun()


@taifun.fn()
def get_location() -> str:
    """
    Get the user's location

    returns: the user's location like a Ciry and State, e.g. San Francisco, CA
    """
    location = Prompt.ask("What is your location?")

    return location


@taifun.fn()
def get_lang_lat(location: str) -> dict:
    """
    Get the latitude and longitude of a location

    Parameters
    ----------
    location: str
        the user's location like a Ciry and State, e.g. San Francisco, CA

    """

    response = httpx.get(
        f"https://nominatim.openstreetmap.org/search/{urlparse.quote(location)}",
        params={
            "format": "json",
        },
    )
    response.raise_for_status()
    data = response.json()
    lat = data[0]["lat"]
    lng = data[0]["lon"]

    return {"latitute": lat, "longitude": lng}


class Coordinates(BaseModel):
    latitude: float = Field(
        ..., title="Latitude", description="The latitude of a location"
    )
    longitude: float = Field(
        ..., title="Longitude", description="The longitude of a location"
    )


@taifun.fn()
def get_current_weather(coordinates: Coordinates):
    """Get the current weather in a given longitude and latitude

    Parameters
    ----------
    coordinates: Coordinates
        the latitude and longitude of a location

    Returns:
        dict: a dictionary of the current weather

    """

    response = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": coordinates.latitude,
            "longitude": coordinates.longitude,
            "current_weather": True,
        },
    )
    response.raise_for_status()
    data = response.json()
    return data


if __name__ == "__main__":
    openai.api_key_path = os.path.expanduser("~") + "/.openai_api_key"
    runner = TaifunConversationRunner(taifun)
    result = runner.run("Will I need an umbrella today?")

    rich.print(result)
