import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "../../")
sys.path.append(project_root)

import requests
import json
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from openai import OpenAI
from recommender.front.recommend import Recommender

app = Flask(__name__)

# Read keys from config

# Define paths to configuration files
current_dir = os.path.dirname(__file__)
configs_path = os.path.join(current_dir, 'keys/config.json')
description_path = os.path.join(current_dir, 'static_data/description.json')

# Read API keys from config files
with open(configs_path) as config_file:
    config = json.load(config_file)

with open(description_path) as description_file:
    descriptions = json.load(description_file)

# Set up the OpenAI API key for ChatGPT
chatgpt_api_key = config['OPEN_AI_KEY']
client = OpenAI(api_key=chatgpt_api_key)

# Yandex Maps API key
yandex_maps_api_key = config['YANDEX_MAPS_KEY']

# Check if Yandex Maps API key and ChatGPT API key are provided
if not yandex_maps_api_key:
    raise ValueError("key 'yandex_maps_api_key' is not provided")
if not chatgpt_api_key:
    raise ValueError("key 'chatgpt_api_key' is not provided")


class GeoPoint:
    # Class attributes
    name = None  # Store the name of the location
    description = "empty"  # Store a description of the location (default is "empty")
    coordinates = None  # Store the geographic coordinates of the location

    # Constructor (__init__) method
    def __init__(self, name=None, location=None, reviews=None):
        # Initialize object attributes with provided values or defaults
        self.name = name  # Name of the location
        self.location = location  # Location information
        self.set_coordinates()  # Set the geographic coordinates
        self.set_description()  # Set the description
        self.reviews = self.total_review(reviews)  # Generate or store reviews

    # Method to set geographic coordinates
    def set_coordinates(self, coordinates=None):
        # If coordinates are not provided, try to fetch them using Yandex Maps API
        if coordinates is None:
            self.coordinates = get_location_coordinates(self.location + " " + self.name)
        else:
            self.coordinates = coordinates

    # Method to set the description
    def set_description(self, description=None):
        # If description is not provided, check if it exists in the descriptions dictionary
        if description is None:
            if self.name in descriptions:
                self.description = descriptions[self.name]['description']
            else:
                self.description = "Description was not found"
        else:
            self.description = description

    # Method to handle reviews (currently not implemented, returns "No reviews")
    # TODO Kate Chuiko (@Kontrosha) implement it
    def total_review(self, reviews):
        # This method could be used to process and summarize reviews
        # Currently, it returns a placeholder message
        return "No reviews"

    # Method to get a dictionary representing the GeoPoint
    def get_point(self):
        # Return a dictionary with location information
        return {
            "name": self.name,  # Location name
            "coordinates": self.coordinates,  # Geographic coordinates
            "description": self.description,  # Location description
            "review": self.reviews  # Reviews or "No reviews"
        }


def get_location_coordinates(name):
    """
    Get the name of a location using Yandex Maps API.
    """
    url = f"https://geocode-maps.yandex.ru/1.x/?apikey={yandex_maps_api_key}&format=json&geocode=Австрия+Тироль+{name.lower()}"
    response = requests.get(url)
    data = response.json()
    try:
        coordinates = data['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point']['pos'].split(
            ' ')
        return coordinates[1], coordinates[0]
    except (KeyError, IndexError):
        return "Unknown Location"


@app.route('/get_closest_coordinates', methods=['POST'])
@cross_origin()
def get_closest_coordinates():
    data = request.get_json()

    # Extract parameters from frontend
    user_hotel = data.get('hotel')

    # Get coordinates from ML based on the user's category
    recommendator = Recommender()
    recommended_places = recommendator.recommend(user_hotel)

    geo_points = [GeoPoint(place['name'], place['location'], place['reviews']) for place in recommended_places]

    return jsonify([point.get_point() for point in geo_points])


if __name__ == '__main__':
    app.run(debug=True, port=10888)
