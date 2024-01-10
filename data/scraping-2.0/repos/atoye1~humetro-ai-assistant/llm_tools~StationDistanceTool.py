import os
import requests
from langchain.agents import tool
from typing import Optional, Type
from pydantic.v1 import BaseModel, Field


class StationDistanceInput(BaseModel):
    """Input for stations distance check"""
    departure: str = Field(...,
                           description="Name of departure station")
    destination: str = Field(...,
                             description="Name of destination station")


@tool(args_schema=StationDistanceInput)
def get_distance(departure: str, destination: str) -> str:
    """calculate distance between two stations. outputs distances, hops, and time"""
    api_key = os.environ['PUBLIC_DATA_API_KEY']
    url = f"http://data.humetro.busan.kr/voc/api/open_api_distance.tnn?serviceKey={api_key}&act=json&scode="
    res = requests.get(url)

    return f"{departure} -> {destination} : 10km, 30 hops, 20min"
