from datetime import date
from fastapi import APIRouter
from configs.configs import settings
from datetime import datetime, timedelta
import requests
from utils.poi_util import get_coordinates_from_address_google_api, get_coordinate_info_from_address_open_street_map

from openai import OpenAI

client = OpenAI(api_key=settings.gpt_key)

router = APIRouter(
    tags=['Trip Weather']
)

@router.get("/api/trip/weather")
def weather_recommendation(location: str, start_date: date, end_date: date):
    print("Inside Weather API.")
    print(location, start_date, end_date)
    current_date = datetime.now().date()
    four_days_from_now = current_date + timedelta(days=4)
    is_forecast_available = False

    if start_date > four_days_from_now:
        print("Weather is not available for the next 4 days.")
        prompt_str = f"Considering past weather conditions in {location} city from {start_date} tp {end_date}, write weather summary, clothing recommendations for the trip, and any other suggestions for the trip.\n "
        gpt_response = gpt(prompt_str)       

        return {"isForecastAvailable": is_forecast_available,"weatherData":gpt_response}
    else:
        is_forecast_available = True
        print("Fetching weather data...")
        lat = lon = 0
        coordinate_info = get_coordinate_info_from_address_open_street_map(location)
        if coordinate_info is not None:
            print("Wow")
            lat, lon = coordinate_info
        else:
            coordinate_info_google = get_coordinates_from_address_google_api(location)
            if coordinate_info_google is not None:
                lat, lon = coordinate_info_google
            else:
                print("Error: Unable to get coordinates for address")
                return {"isForecastAvailable": False,"weatherData":"Weather data is not available"}

        url = "https://api.openweathermap.org/data/3.0/onecall?&exclude=minutely,hourly"
        payload = {"lat": lat, "lon": lon, "appid": settings.openweather_api_key}

        weather_data = requests.request("GET", url, params=payload).json()
        main_weather_icon = weather_data['current']['weather'][0]['icon']
        print("Weather data fetched successfully.", main_weather_icon)
        filtered_data = [weather_data['current']]

        for forecast in weather_data['daily']:
            forecast_date = datetime.utcfromtimestamp(forecast['dt']).date()
            if start_date <= forecast_date <= end_date:
                filtered_data.append(forecast)
        
        prompt_str = f"Given weather forecast data for the dates in {location} city, return weather summary, clothing recommendations for the trip, and any other suggestions for the trip.\n The weather forecast data is as follows: {filtered_data}. Return temperature in Fahrenheit.\n"
        gpt_response = gpt(prompt_str)

        return {"isForecastAvailable": is_forecast_available,"weatherData":gpt_response, "main_weather_icon": main_weather_icon}

def gpt(prompt_str):
    response = client.chat.completions.create(
        model= settings.gpt_model,
        messages=[
            {"role": "user", "content": prompt_str}
        ]
    )
    res = str(response.choices[0].message.content)
    return res
