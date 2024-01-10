from pydantic import Field, BaseModel
from langchain.tools import BaseTool
from typing import Literal

class GetCurrentWeatherInput(BaseModel):
    location: str = Field(description="天気を知りたい場所を入力してください")

class WeatherModel(BaseModel):
    location: str
    temperature: float
    unit: Literal["celsius", "fahrenheit"]
    forecast: Literal["sunny", "windy"]
