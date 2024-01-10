import json
from enum import Enum
from typing import Literal

from openai_func_call.manager import FunctionManager


class TemperatureUnit(str, Enum):
    """A temperature unit."""

    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location: str, unit: TemperatureUnit = "fahrenheit") -> str:
    """Get the current weather in a given location
    :param location: The city and state, e.g. San Francisco, CA
    :param unit: The unit to return the temperature in
    """
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


def get_current_weather2(location: str, unit: Literal["fahrenheit", "celsius"] = "fahrenheit") -> str:
    """Get the current weather in a given location
    :param location: The city and state, e.g. San Francisco, CA
    :param unit: The unit to return the temperature in
    """
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


def run_conversation():
    # Step 1: send the conversation and available functions to GPT
    query = "What's the weather like in Boston, in EU units, and what is it like in Miami?"
    return FunctionManager([get_current_weather]).query_openai(query)


print(run_conversation())
