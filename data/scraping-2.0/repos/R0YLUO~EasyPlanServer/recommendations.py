import googlemaps
import json 
import os
from dotenv import load_dotenv
import openai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

map_client = googlemaps.Client(GOOGLE_API_KEY)
openai.api_key = OPEN_AI_KEY

def generate_recommendations(location: tuple[float,float], cuisine_types : list[str], radius: int):
    '''

    :param location: center location as tuple (lat,long)
    :param cuisine_type: cuisine types as strings in a list
    :param radius: radius of search, smaller radius is closer to center but larger radius gets better results
    :return: JSON object of recommendations
    '''

    response = {
        "results": []
    }
    for cuisine in cuisine_types:
      res = map_client.places_nearby(
          location = location,
          keyword = cuisine,
          radius = radius
      )
      print(type(res.get('results')))
      response['results'].append({
        "cuisine": cuisine,
        "result": get_first_five_elements(res.get('results'))
      })

    return json.dumps(response)

def get_first_five_elements(my_list):
    if len(my_list) >= 5:
        return my_list[:5]
    else:
        return my_list

def find_center(location_list: list[tuple]) -> tuple:
    '''

    :param location_list: list of locations as tuples (lat, long)
    :return: center of these locations
    '''

    n = len(location_list)

    sum_x = 0
    sum_y = 0

    for location in location_list:
        sum_x += location[0]
        sum_y += location[1]

    result = (sum_x/n,sum_y/n)

    return result

def find_top_food_preference(food_preferences: list[str]):
    prompt = "Given the following the food preferences, tell me top 3 most popular food preferences by putting them into an array of singular words describing the cuisine of preference and only returning that array. If there's not enough for 3, give as many as possible. \n\n" + "\n".join(food_preferences)
    
    response = openai.ChatCompletion.create(
      model = "gpt-3.5-turbo",
      temperature = 0.2,
      max_tokens = 1000,
      messages = [
        {"role": "user", "content": prompt}
      ]
    )



    return json.loads(response['choices'][0]['message']['content'])
      
def recommendation(location_list: list[list], food_preferences: list[str]):
    return generate_recommendations(find_center(location_list),find_top_food_preference(food_preferences),30)

test_locations= [(-37.81318596153627, 144.96377579645537), (-37.78519512418844, 144.7710836503532), (-37.861495369958895, 145.11253287046154)]