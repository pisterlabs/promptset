from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.tools import Tool

weather = OpenWeatherMapAPIWrapper()
weather_tool = Tool.from_function(
        func=weather.run,
        name="weather",
        description="Use this tool only to search for any weather related information, the default location is san jose, california. Give a short description",
        return_direct=True
    )
