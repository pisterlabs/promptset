from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from ..tools.search import reverse_geocoding


class ReverseGeodecodeSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    longitude: float = Field(description="Longitude of the location")
    latitude: float = Field(description="Latitude of the location")


class ReverseGeodecodeTool(BaseTool):
    name = "reverse_geodecode"
    description = ('''
    This function transforms the geo coordinates(longitude and latitude) to POIs (mainland China).
    The POI correspond to the information of the location.
    The usage of this tool is strictly limited.
    Please limit your usage to avoid exceeding the limit.(use geo_decode instead)
    ''')
    args_schema: Type[ReverseGeodecodeSchema] = ReverseGeodecodeSchema

    def _run(
            self,
            longitude: float,
            latitude: float,
    ):
        """Use the tool."""
        return reverse_geocoding.reverse_geocoding(longitude, latitude)
