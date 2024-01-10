from fastapi import APIRouter
from openai import OpenAI

import string
from configs.db import db
from configs.configs import settings
from utils import poi_util, user_util, prompt_util, file_util
from typing import Optional

from utils.geo_hash_util import get_nearby_poi_ids


client = OpenAI(api_key=settings.gpt_key)
collection_poi = db['poi']

router = APIRouter(
    tags=['GPT Recommender']
)

@router.post("/api/get/gpt/recommendation")
def generate_recommendation( user_name: str, city_id: Optional[str] = None, city_name: Optional[str] = None):
    print("Generating training data for recommendation of places in city : ", city_name)
    print("City id: ", city_id)
    
    collection_city = db['city']
    if city_id != None:
        city = collection_city.find_one({'city_id': city_id})
    else:
        destination = string.capwords(city_name)
        city = collection_city.find_one({'city_name': destination})
    destination = city['city_name']

    gpt_output = generate_recommnedation_from_city_object(city, user_name)

    return gpt_output

# This function is used to generate presonalized description for a place using gpt model based on a particular user preferences.
@router.post("/api/gpt/personalized/description")
def generate_recommendation(user_name : str, poi_id: int, city_id: Optional[str] = None, city_name: Optional[str] = None):
    
    collection_city = db['city']
    if city_id != None:
        city = collection_city.find_one({'city_id': city_id})
    else:
        destination = string.capwords(city_name)
        city = collection_city.find_one({'city_name': destination})
    destination = city['city_name']
    poi_list = city['pois']
    poi = poi_util.get_poi_by_id_from_poi_list(poi_list, poi_id)

    preferences = user_util.get_user_preferences(user_name)
    gpt_prompt = prompt_util.generate_gpt_prompt_for_personalized_description(destination, preferences, poi)
    response = client.chat.completions.create(
        model= settings.gpt_model,
        messages=[
            {"role": "user", "content": gpt_prompt}
        ]
    )
    gpt_output = str(response.choices[0].message.content)
    
    return gpt_output

# This function is used to generate presonalized description for a place using gpt model based on a particular user preferences.
@router.post("/api/gpt/personalized/description_1")
def generate_recommendation_1(user_name : str, poi_id: int):
    poi = poi_util.get_poi_from_poi_id(poi_id)
    preferences = user_util.get_user_preferences(user_name)
    destination = poi['city']
    gpt_prompt = prompt_util.generate_gpt_prompt_for_personalized_description(destination, preferences, poi)
    response = client.chat.completions.create(
        model= settings.gpt_generic_model,
        messages=[
            {"role": "user", "content": gpt_prompt}
        ]
    )
    gpt_output = str(response.choices[0].message.content)
    return gpt_output

def generate_recommnedation_from_city_object(city, user_name):
    preferences = user_util.get_user_preferences(user_name)
    poi_list = city['pois']
    destination = poi_list[0]["city"]
    gpt_prompt = prompt_util.generate_gpt_prompt(destination, preferences, poi_list)
    # print(gpt_prompt)
    response = client.chat.completions.create(
        model= settings.gpt_model,
        messages=[
            {"role": "system", "content": "You are a helpful recommendation engine which returns the array of ids of recommended places based on the given preferences. If possible it will suggest 5 places i.e output will be array of size 5"}, 
            {"role": "user", "content": "Create recommendations for a user based on preferences : ['Museum'] for the places in city :San Jose from the follwoing array of point_of_intrests : point_of_intrests in [ name : Rosicrucian Egyptian Museum id : 47435 type of place : [Museum,Archaeological museum,History Museums,Specialty Museums,] name : San Pedro Square Market Bar id : 112876 type of place : [Bar,Shopping,Flea & Street Markets,]]"},
            {"role": "assistant", "content": "[47435]"},
            {"role": "user", "content": gpt_prompt}
        ]
    )
    gpt_output = str(response.choices[0].message.content)
    gpt_prompt += 'gpt_response : ' + gpt_output
    file_util.write_string_to_file("prompt.josnl", gpt_prompt)

    return gpt_output

@router.post("/api/get/gpt/runtime/recommendation")
def get_runtime_recommendations(user_input: str, address: str, radius: int):
    latitude, longitude = poi_util.get_coordinates_from_address(address)
    nearby_poi_ids = get_nearby_poi_ids(latitude, longitude, radius)
    if len(nearby_poi_ids) == 0:
        print("No nearby pois found")
    poi_list = []
    for poi_id in nearby_poi_ids:
        poi = collection_poi.find_one({'poi_id': poi_id} , {'_id': 0})
        if poi != None:
            poi_list.append(poi)
    gpt_prompt = prompt_util.generate_gpt_prompt_for_runtime_recommendation_with_address(user_input, poi_list)
    response = client.chat.completions.create(
        model= settings.gpt_model,
        messages=[
            {"role": "user", "content": gpt_prompt}
        ]
    )
    gpt_output = str(response.choices[0].message.content)
    file_util.write_string_to_file("runtime_gpt_prompts.jsonl", gpt_prompt + "gpt_response : " + gpt_output)
    return gpt_output