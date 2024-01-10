"""This module contains the WeatherTool class."""
from typing import Optional

import requests
from langchain.tools import BaseTool


class WeatherTool(BaseTool):
    """This tool allows you to get the weather using the wttr.in service."""

    name = "Weather"
    description = (
        "This tool allows you to get the weather using the wttr.in service;"
        "you can get the weather in your current location by passing no "
        "arguments or in a specific location by passing the location as an "
        "argument;"
    )

    def _run(self, *args, query: Optional[str] = None, **kwargs) -> str:
        if query is None:
            query = ""

        try:
            response = requests.get(f"https://wttr.in/{query}?format=4", timeout=5)
        except requests.exceptions.Timeout:
            return "Sorry, the weather service is not responding right now."
        except requests.exceptions.ConnectionError:
            return "Sorry, could not connect to the weather service."
        except requests.exceptions.RequestException:
            return "Sorry, something went wrong with the weather service."

        return response.text

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError("Weather does not support async")
