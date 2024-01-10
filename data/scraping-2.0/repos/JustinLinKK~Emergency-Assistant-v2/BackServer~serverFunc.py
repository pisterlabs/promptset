import openai
import googlemaps
from datetime import datetime

def audio_to_text(audio_file_name, openai_api_key):
    
    openai.api_key = openai_api_key
    audio_target = open(audio_file_name, "rb")
    result = openai.Audio.transcribe("whisper-1",audio_target)
    return result

def reclassify(input_text,prefix,openai_api_key):
    '''
    This function takes an input text and reclassifies it.
    Input: input_text: Text to be reclassified.
            prefix: Prefix to be used for reclassification.
            api_key: OpenAI API key.
    Output: Reclassified text.        
    '''
    openai.api_key = openai_api_key
    messages = [{"role": "system", "content": prefix}]    
    messages.append({"role": "user", "content": input_text},)    
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)    
    text = str(chat.choices[0].message.content)
    return text

def find_nearest_services(api_key, address):
    # Initialize the Google Maps client
    gmaps = googlemaps.Client(key=api_key)

    # Geocode the address to get latitude and longitude
    geocode_result = gmaps.geocode(address)
    if not geocode_result:
        return "Address not found."

    location = geocode_result[0]['geometry']['location']

    # Define the types of places we are looking for
    place_types = ['fire_station', 'hospital', 'police']

    # Store the results
    results = {}

    for place_type in place_types:
        # Find the nearest place of the specified type
        places_result = gmaps.places_nearby(location=location, type=place_type, rank_by='distance')

        if places_result['results']:
            # Take the closest place
            closest_place = places_result['results'][0]
            place_name = closest_place['name']
            place_location = closest_place['geometry']['location']

            # Calculate the distance and time to the place
            distance_result = gmaps.distance_matrix(origins=location,
                                                    destinations=place_location,
                                                    mode="driving",
                                                    departure_time=datetime.now())

            # Extract distance and duration
            distance_info = distance_result['rows'][0]['elements'][0]
            distance = distance_info['distance']['text']
            duration = distance_info['duration']['text']

            # Store the information
            results[place_type] = {'name': place_name, 'distance': distance, 'duration': duration}

    return results

def get_location(api_key, address):
    # Initialize the Google Maps client
    gmaps = googlemaps.Client(key=api_key)

    # Geocode the address
    geocode_result = gmaps.geocode(address)

    # Check if any results were returned
    if not geocode_result:
        return "Address not found."

    # Extract latitude and longitude
    location = geocode_result[0]['geometry']['location']
    latitude = location['lat']
    longitude = location['lng']
    
    return latitude, longitude


