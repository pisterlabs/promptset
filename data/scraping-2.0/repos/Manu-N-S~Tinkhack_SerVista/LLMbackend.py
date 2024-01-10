import openai
from geopy.geocoders import Nominatim
import requests

# from flask import Flask,request,jsonify
# app = Flask(__name__)

# @app.r

# Set your OpenAI GPT-3 API key
openai.api_key = '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'

current_item=""
main_prompt =""
food_items =""
hotel_names=""
current_location = ""
def find_nearest_airport(latitude, longitude):
    # OpenSky Network API URL for fetching nearby airports
    url = f'https://opensky-network.org/api/airports/near?lat={latitude}&lng={longitude}&max=1'

    # Make the API request
    response = requests.get(url)
    data = response.json()

    # Extract and print airport details
    if data:
        airport = data[0]
        print(f'Nearest Airport: {airport["name"]}, ICAO Code: {airport["icao"]}')
    else:
        print('No nearby airports found.')

def chat_with_gpt(prompt):
    global current_item
    try:
        # Make a request to the OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-002",  # Specify the GPT-3 engine
            prompt=prompt,
            max_tokens=150  # You can adjust the maximum number of tokens in the response
        )

        # Print the generated response
        current_item = response['choices'][0]['text']
        print(current_item)
        
    except Exception as e:
        print(f"Error: {e}")

def get_fooditemprompt(prompt):
    try:
        # Make a request to the OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-002",  # Specify the GPT-3 engine
            prompt=prompt,
            max_tokens=150  # You can adjust the maximum number of tokens in the response
        )

        # Print the generated response
        food_items = response['choices'][0]['text']
        print(f"Food items are:{food_items}")
        
    except Exception as e:
        print(f"Error: {e}")

def get_hotelprompt(prompt):
    try:
        # Make a request to the OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-002",  # Specify the GPT-3 engine
            prompt=prompt,
            max_tokens=150  # You can adjust the maximum number of tokens in the response
        )

        # Print the generated response
        hotel_names = response['choices'][0]['text']
        print(f"hotel mentioned are:{hotel_names}")
        
    except Exception as e:
        print(f"Error: {e}")

def get_destination(prompt):
    global travel_destination
    try:
        # Make a request to the OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-002",  # Specify the GPT-3 engine
            prompt=prompt,
            max_tokens=150  # You can adjust the maximum number of tokens in the response
        )

        # Print the generated response
        travel_destination = response['choices'][0]['text']
        print(f"Travel Destination:{travel_destination}")
        
    except Exception as e:
        print(f"Error: {e}")

def get_currentLoc():
    #get it from MAP API
    pass

def get_nearAirport(destination,current):
    geolocator = Nominatim(user_agent="airport_finder")
    location = geolocator.geocode(destination)
    if location:
        latitude, longitude = location.latitude, location.longitude
        print(f'Your Location - Latitude: {latitude}, Longitude: {longitude}')

        # Find the nearest airport
        find_nearest_airport(latitude, longitude)
    else:
        print('Location not found.')

# Interactive chat loop
print("Welcome to the GPT-3 Chat! Type 'exit' to end the conversation.")
while True:
    front_end_text = input("You: ")
    
    if front_end_text.lower() == 'exit':
        break

    # Incorporate user input into the prompt
    # prompt = f"User: {user_input}\nGPT-3:"
    
    main_prompt = front_end_text
    user_input = "what is this refering to ? FOOD,TRAVEL,EXPENSE give it in only one word"
    prompt = f"User: {front_end_text}{user_input}\nGPT-3:"
    # Get GPT-3 response
    chat_with_gpt(prompt)
    print(current_item)
    if "Food" in current_item:
        food_search_data = "what is the Food items mentioned ? give names only seperated by comma"
        prompt = f"User: {front_end_text}{food_search_data}\nGPT-3:"
        get_fooditemprompt(prompt)
        hotel_search = "what is the name of the restaurant mentioned ? give only names seperated by comma"
        prompt = f"User: {front_end_text}{hotel_search}\nGPT-3:"
        get_hotelprompt(prompt)
    elif "Travel" in current_item:
        destination = "what is destination location name ? Give it in one word only"
        prompt = f"User: {front_end_text}{destination}\nGPT-3:"
        get_destination(prompt)
        current_location = get_currentLoc()
        find_nearAirport(destination,curr)


