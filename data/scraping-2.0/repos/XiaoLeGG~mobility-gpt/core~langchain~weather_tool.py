from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from ..tools.search import historical_weather


class WeatherSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    longitude: float = Field(description="Longitude of the location")
    latitude: float = Field(description="Latitude of the location")
    start_date: str = Field(description="The start date of the historical weather in the format of 'YYYY-MM-DD'")
    end_date: str = Field(description="The end date of the historical weather in the format of 'YYYY-MM-DD'")


class WeatherTool(BaseTool):
    name = "historical_weather"
    description = ('''
    This function searches the historical weather from openmeteo API.
    ''')
    args_schema: Type[WeatherSchema] = WeatherSchema

    def _run(
            self,
            longitude: float,
            latitude: float,
            start_date: str,
            end_date: str,
    ):
        """Use the tool."""
        return historical_weather.historical_weather(longitude, latitude, start_date, end_date)
