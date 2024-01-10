import requests
import environ
import openai
from django.utils import timezone

from apps.weather.models import WeatherData, ClothingRecommendations
from config.settings.base import BASE_DIR

from typing import Optional, Dict


TOKENS_TO_EXTRACT = {"clothing": "Clothing: ", "shoes": "Shoes: ", "accessories": "Accessories: "}

env = environ.Env()
env.read_env(str(BASE_DIR / ".env"))


def call_weather_api(city: str) -> dict:
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={env('API_KEY')}&units=metric"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        weather_data = response.json()

        # Extract weather data
        weather_details = weather_data['weather'][0]
        main_details = weather_data['main']
        wind_details = weather_data['wind']
        sys_details = weather_data["sys"]

        weather_info = {
            "city_name": weather_data['name'],
            "weather_main": weather_details['main'],
            "temperature": round(main_details['temp']),
            "humidity": main_details['humidity'],
            "wind_speed": wind_details['speed'],
            "weather_icon": weather_details['icon'],
            "country": sys_details["country"]
        }
        return weather_info
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None


def get_clothing_recommendation(weather_data: dict) -> Optional[Dict[str, str]]:
    openai.api_key = env('OPENAI_KEY')

    prompt = f"""Given the weather data: {weather_data}, recommend suitable clothing in the following form (3 categories):
    Clothing: maximum 5 elements of suitable clothing
    Shoes: maximum 5 elements of suitable clothing
    Accessories: maximum 3 elements of suitable accessories
    """

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )

        return parse_clothing_suggestions(response.choices[0].text)

    except Exception as e:
        print(f"Error generating clothing recommendation: {e}")
        return None


def extract_suggestion(token: str, suggestion_text: str) -> Optional[str]:
    if token in suggestion_text:
        return suggestion_text.replace(token, "")


def parse_clothing_suggestions(closing_suggestions: str) -> Optional[Dict[str, str]]:
    extracted_suggestions = {}
    for token_name, token_value in TOKENS_TO_EXTRACT.items():
        for suggestion_text in closing_suggestions.split("\n"):
            extracted_suggestion = extract_suggestion(
                token=token_value, suggestion_text=suggestion_text
            )
            if extracted_suggestion:
                extracted_suggestions[token_name] = extracted_suggestion
    return extracted_suggestions


def save_weather_data(api_response) -> WeatherData:
    weather_info = api_response

    # Create a new WeatherData instance
    new_weather_data = WeatherData(
        city_name=weather_info['city_name'],
        weather_main=weather_info['weather_main'],
        temperature=weather_info['temperature'],
        humidity=weather_info['humidity'],
        wind_speed=weather_info['wind_speed'],
        weather_icon=weather_info['weather_icon'],
    )

    new_weather_data.created_at = timezone.now()
    new_weather_data.modified_at = timezone.now()

    # Save
    new_weather_data.save()

    return new_weather_data


def save_clothing_recommendations(weather_data_instance: WeatherData, description_text: dict) -> None:

    # Create a ClothingRecommendations instance.
    weather_description = ClothingRecommendations(weather_data=weather_data_instance, description=description_text)

    weather_description.save()
