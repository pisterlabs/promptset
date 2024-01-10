from typing import Union

import requests
from langchain.tools import StructuredTool
from pydantic import BaseModel

from common.base_thread import get_logger


class WeatherInfo(BaseModel):
    country: str
    city: str
    wind: str
    wind_speed: str
    air_level: str
    temperature_now: str
    temperature_highest: str
    temperature_lowest: str


# class GetWeatherInfoInChinaParameters(BaseModel):
#     location: str = Field(description="should be a city in china")
#
#
# @tool("get_weather_info_in_china", return_direct=True, args_schema=GetWeatherInfoInChinaParameters)
def get_weather_info_in_china(location: str) -> Union[WeatherInfo, str]:
    """
    useful for when you need to get the current weather information only in China
    firstly, you must know which country is the location of,
    if it's china call this tool,
    but if it isn't, do not call.
    secondly, you must make sure the location is a valid city
    """
    try:
        get_logger().info(f"location of the request is {location}")
        resp = requests.get(
            f'http://www.tianqiapi.com/api?version=v6&appid=23035354&appsecret=8YvlPNrz&city={location}')
        resp.encoding = 'utf-8'
        content = resp.json()

        wi = WeatherInfo(
            city=content['city'],
            country=content['country'],
            wind=content['win'],
            wind_speed=content['win_speed'],
            air_level=content['air_level'],
            temperature_now=content['tem'],
            temperature_lowest=content['tem2'],
            temperature_highest=content['tem1'])

        get_logger().info(f"response {wi}")
        return wi
    except Exception as e:
        get_logger().error(e)
        return "get_weather failed because of bad network, maybe tell the user to shutoff the proxy will be better"


get_weather_info_in_china = StructuredTool.from_function(get_weather_info_in_china)
