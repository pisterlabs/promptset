
import string

import openai
from bson.objectid import ObjectId
from fastapi import APIRouter
from pydantic import BaseModel

import utils.file_util as file_util
import utils.ai_util as ai_util
from configs.configs import settings
from configs.db import db
from models.trip import Trip
from models.user import User
import json
from typing import Optional
from utils.geo_hash_util import get_nearby_poi_ids
from utils.poi_util import get_coordinates_from_address
from utils.prompt_util import generate_gpt_prompt_for_runtime_recommendation, generate_gpt_prompt_for_runtime_recommendation_with_address

openai.api_key = settings.gpt_key


router = APIRouter(
    tags=['Fine Tuning']
)
collection_poi = db['poi']

# The function below is used to generate training data for gpt model.
@router.post("/api/finetune/generate/recommendation")
def generate_recommendation( preferences: list, city_id: Optional[str] = None, city_name: Optional[str] = None):
    print("Generating training data for recommendation of places in city")
    print("City id: ", city_id)
    print("City name: ", city_name)
    print("Preferences: ", preferences)
    collection_city = db['city']
    if city_id != None:
        city = collection_city.find_one({'city_id': city_id})
    else:
        destination = string.capwords(city_name)
        city = collection_city.find_one({'city_name': destination})
    destination = city['city_name']
    poi_list = city['pois']

    filtered_poi_list = ai_util.get_filtered_poi_list(poi_list, preferences)
    # print("Filtered poi list: ", filtered_poi_list)

    poi_names = []
    
    for poi in poi_list:
        if poi['type'] != None:
            type_string = ""
            types = poi['type']
            for type1 in types:
                type_string += type1 + ", "
            poi_names.append({"name" :poi['name'], "id": poi['poi_id'] , 
                              "type" : type_string
                            })
    
    training_propmt = '{"messages": [{"role": "system", "content": "Marv is a recommendation engine which is able to recommend tourist places to users based on the given preferences."}, {"role": "user","content": '
    training_propmt += '"Create recommendations for a user based on preferences : ' + str(preferences)
    training_propmt += ' for the places in city :' + destination
    training_propmt += ' from the follwoing array of point_of_intrests : point_of_intrests in [' 
    for poi in poi_names:
        training_propmt +=  ' name : ' + poi['name'] + ' id : ' + str(poi['id']) + ' type of place : ' + poi['type'] +','
    training_propmt += ']"}'
    training_propmt += ' , {"role": "assistant", "content": "[418371, 47435, 28163, 112861, 112897]"}]}'

    request = {
        "preferences": preferences,
        "city": city_name,
        "poi_names": poi_names,
    }
    prompt = "Create recommendations for the tourists places based on preferences from the given request object. Request: " + str(request) + "\n\nResponse format: [p,q,r,s,t]. Response is list of ids where each id is the id of place of poi_names that is recommended to the user based on preferences. This list will contain top 5 recommendations."
    print('-------------------------------------------------------------------------')
    print('token lenght of prompt', ai_util.get_token_length(prompt))
    print('token lenght of training prompt', ai_util.get_token_length(training_propmt))
    print('-------------------------------------------------------------------------')
    print('prompt', prompt)
    print('-------------------------------------------------------------------------')
    file_util.write_string_to_file("trip_ai_train.josnl", training_propmt)

    # print(training_propmt)
    # response = openai.Completion.create(
    #     engine="gpt-3.5-turbo-instruct",
    #     prompt=prompt,
    #     temperature=0.7,
    #     max_tokens=50,
    # )
    # itinerary = response.choices[0].text
    # return itinerary

# The function below will be used to trigger the training job for gpt 
@router.post("/api/finetune/upload/training/file/{upload_file_name}")
def upload_training_file(upload_file_name):
    print("-------------------------------------------------------")
    # upload_file_name = "train_gpt.josnl"
    print("Uploading file: ", upload_file_name)
    file_key = openai.File.create(
        file=open(upload_file_name, "rb"),
        purpose="fine-tune"
        )
    print("file_key : ", file_key)
    return file_key

# This function will be used to create a finetuining job for gpt model
@router.post("/api/finetune/create/job/{file_key}")
def create_trip_recommendation(file_key : str):
    print("Starting training job for gpt model")
    return openai.fine_tuning.job.create(
        training_file=file_key, 
        model="gpt-3.5-turbo"
    )

# @router.post("/api/trip/create/recommendation")
# def create_trip_recommendation(trip_data: TripCreation):
#     # Define a list of user preferences questions
#     # user_preference_array = ["mountains", "quiet", "peaceful", "parks"]
#     user_preference_array = ['park']
#     destination = "San Jose"

#     # user_preference_array = [
#     # # "User is somewhat adventurous when trying new activities or experiences.",
#     # "User wants to go for hiking",
#     # # "User enjoys a balance of both indoor and outdoor activities when traveling.",
#     # # "User enjoys natural landscapes like mountains the most.",
#     # # "User is interested in historical and cultural attractions.",
#     # # "User has a keen interest in photography during their trips.",
#     # ]
    
#     destination = string.capwords(destination)
#     collection_city = db['city']
#     city = collection_city.find_one({'city_name': destination})
#     poi_list = city['pois']
#     poi_names = []
    
#     for poi in poi_list:
#         if poi['type'] != None:
#             poi_names.append({"poi_name" :poi['name'], "id": poi['poi_id'] , 
#                             #   "type" : poi['type']
#                             })
    
#     training_propmt = '{"messages": [{"role": "system", "content": "Marv is a recommendation engine which is able to recommend tourist places to users based on the given preferences."}, {"role": "user","content": '
#     training_propmt += '"Create recommendations for a user based on preferences : ' + str(user_preference_array)
#     training_propmt += 'for the places in city :' + destination
#     training_propmt += 'from the follwoing array of point_of_intrests : "point_of_intrests" : ' + str(poi_names) + '}'
#     training_propmt += ' , {"role": "assistant", "content": "[418371, 47435, 28163, 112861, 112897]"}]}'

#     request = {
#         "preferences": user_preference_array,
#         "city": city,
#         "poi_names": poi_names,
#     }

#     # Generate a prompt for GPT-3.5 based on user responses
#     # prompt="From the give list of poi_names give me the destinations user want to vist the most based on the user preference and type of place. Request: " + str(request) + "\n\nResponse format: [p,q,r,s....]. Response is list of poi_ids where each element in the list is a point of interest. here the point of intrest are in sorted order that means the user will visit point of interest p then q then r and so on"

#     # prompt = "Create recommendations for the tourists places based on preferences from the given request object. Request: " + str(request) + "\n\nResponse format: [p,q,r,s....]. Response is list of ids where each id is the id of place of poi_names that is recommended to the user based on preferences."

#     prompt = "Create recommendations for the tourists places based on preferences from the given request object. Request: " + str(request) + "\n\nResponse format: [p,q,r,s,t]. Response is list of ids where each id is the id of place of poi_names that is recommended to the user based on preferences. This list will contain top 5 recommendations."

#     file_util.write_string_to_file("trip_ai_train.josnl", training_propmt)

#     print(training_propmt)
#     # Make a request to GPT-3.5 to generate recommendations
#     response = openai.Completion.create(
#         engine="gpt-3.5-turbo-instruct",
#         prompt=prompt,
#         temperature=0.7,
#         max_tokens=50,
#     )

#     itinerary = response.choices[0].text
#     return itinerary

# # This function will be used to create a finetuining job for gpt model for runtime recommendation based on user input
@router.post("/api/finetune/generate/traindata")
def create_training_prompt_for_runtime_recommencations(user_input_string, address, radius : int):
    print("Creating training prompt for runtime recommendation")
    print("User input string: ", user_input_string)

    latitude, longitude = get_coordinates_from_address(address)
    nearby_poi_ids = get_nearby_poi_ids(latitude, longitude, radius)
    if len(nearby_poi_ids) == 0:
        print("No nearby pois found")
    poi_list = []
    for poi_id in nearby_poi_ids:
        poi = collection_poi.find_one({'poi_id': poi_id} , {'_id': 0})
        if poi != None:
            poi_list.append(poi)
    
    training_propmt = '{"messages": [{"role": "system", "content": "Marv is a recommendation engine which is able to recommend tourist places to users based on the given user input string."}, {"role": '
    training_propmt += '"user","content": '
    # training_propmt += generate_gpt_prompt_for_runtime_recommendation(user_input_string, poi_list)
    user_prompt = generate_gpt_prompt_for_runtime_recommendation_with_address(user_input_string, poi_list)
    training_propmt += user_prompt
    training_propmt += '}'
    training_propmt += ' , {"role": "assistant", "content": "[418371, 47435, 28163, 112861, 112897]"}]}'
    print('-------------------------------------------------------------------------')
    file_util.write_string_to_file("runtime_train.jsonl", training_propmt)
    file_util.write_string_to_file("user_prompt.josnl", user_prompt)

    return {"data" : poi_list}
