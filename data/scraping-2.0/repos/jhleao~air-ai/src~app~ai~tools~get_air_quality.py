from datetime import datetime, timedelta
from langchain.tools import BaseTool
from ...weatherapi.api import OpenWeatherMapApi
from ...schema import AirDataPoint
from typing import List
from ...schema import PlaceNotFoundError
import json


class GetAirQualityTool(BaseTool):
    collected_data: dict[str, List[AirDataPoint]] = {}

    def __init__(self, collected_data: dict[str, List[AirDataPoint]] = {}):
        super().__init__(
            name="get-air-quality",
            description=""""
            Use this to find air quality measurements of one single place.
            Only ask for one place's information at a time.
            Only provide valid location names.
            Invalid location name: "Human's city"
            Valid location name: "New York"
            Input must be JSON formatted with one property called "place" and one called "date", in ISO-8601 format.
            """,
        )

        self.collected_data = collected_data

    def _run(self, query: str) -> str:
        raise NotImplementedError(f"{self.name} does not support sync execution")

    async def _arun(self, input: str) -> str:
        owm = OpenWeatherMapApi()
        parsed_input = json.loads(input)
        place_name = parsed_input.get("place")
        date_iso = parsed_input.get("date")

        if not place_name or not date_iso:
            return "Please provide a place name and date in ISO-8601 format."

        # Fallback to today in an invalid date case.
        try:
            date = datetime.fromisoformat(date_iso)
        except ValueError:
            date = datetime.now()

        date_minus_one_week = date - timedelta(days=7)

        start = int(date_minus_one_week.timestamp())
        end = int(date.timestamp())

        lat_lng = await owm.geocode(place_name)
        points = await owm.air_quality(lat_lng, start, end)

        if len(points) == 0:
            raise PlaceNotFoundError(place_name)

        self.collected_data[place_name] = points

        latest_point = points[-1]

        formatted_date = date.strftime("%B %d, %Y")

        return (
            f"Air quality for {place_name} on {formatted_date}: {latest_point.aqi} AQI"
        )
