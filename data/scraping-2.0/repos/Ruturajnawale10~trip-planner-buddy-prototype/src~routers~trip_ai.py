from datetime import date
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.trip import Trip
from models.user import User
from configs.db import db
from bson.objectid import ObjectId
from configs.configs import settings

import openai

openai.api_key = settings.gpt_key

router = APIRouter(
    tags=['Trip AI generated']
)

class TripCreation(BaseModel):
    createdBy: str
    startDate: date
    endDate: date
    pois: list
    cityName: str

@router.post("/api/trip/create/ai")
def create_trip_own(trip_data: TripCreation):
    request = {
        "preferences": ["mountains", "quiet", "peaceful", "parks"],
        "city": "San Jose",
        "poi_names": [
            {"poi_name": "Rosicrucian Egyptian Museum", "poi_id": 1},
            {"poi_name": "Winchester Mystery House", "poi_id": 2},
            {"poi_name": "Municipal Rose Garden", "poi_id": 3},
            {"poi_name": "Cathedral Basilica of St. Joseph", "poi_id": 4},
            {"poi_name": "PayPal Park", "poi_id": 5}
        ],
        "no_of_days": 2
    }

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt="Create an itinerary based on user preferences and points of interest. Request: " + str(request) + "\n\nResponse format: [[p,q], [r,s]]. Response is list of lists of poi_ids where each list is a day and each element in the list is a point of interest. [p,q] means that on day 1, the user will visit point of interest p and q.",
        temperature=0.7,
        max_tokens=50,
    )

    itinerary = response.choices[0].text
    return itinerary
