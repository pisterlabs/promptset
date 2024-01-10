import requests
from dotenv import load_dotenv
import os
from speech_processing import SpeechProcessing
from openai_agent import OpenAIAgent

load_dotenv()

class WeatherAgent:
    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.base_url = "http://api.weatherapi.com/v1/current.json"
        self.openai_agent = OpenAIAgent()
        self.speech_processor = SpeechProcessing()

    def handle_command(self, command):
        location = self.openai_agent.extract_information("location", command)
        if location == "none":
            self.get_location()
        else:
            weather_data = self.get_weather(location)
            self.process_weather(weather_data)

    def get_location(self):
        self.speech_processor.speak("Please specify a location for me to give you the current weather.")
        location = self.speech_processor.listen()
        location = self.openai_agent.extract_information("location", location)

        if location and location != "none":
            weather_data = self.get_weather(location)
            self.process_weather(weather_data)
        else:
            self.speech_processor.speak("I can't find the specified location. Please try again.")

    def process_weather(self, data):
        if data:
            weather_message = f"Currently in {data['location']}, the weather condition is : {data['condition']}, and the temperature is : {data['temperature']} degrees."
            self.speech_processor.speak(weather_message)
        else:
            self.speech_processor.speak("I couldn't retrieve the weather informations. Please try again.")

    def get_weather(self, location):
        params = {
            'key': self.api_key,
            'q': location,
            'aqi': 'no'
        }

        response = requests.get(self.base_url, params=params)

        if response.status_code != 200:
            return None

        data = response.json()

        weather_data = {
            'location': data['location']['name'],
            'condition': data['current']['condition']['text'],
            'temperature': data['current']['temp_c']
        }

        return weather_data