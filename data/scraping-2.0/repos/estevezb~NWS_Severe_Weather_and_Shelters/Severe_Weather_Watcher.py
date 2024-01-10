from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import requests
from datetime import datetime
from dronekit import connect
import threading
import time
import openai
import os
from geopy.geocoders import Nominatim
from time import sleep
import json






def load_api_key(env_var_name):
    api_key = os.getenv(env_var_name)
    if api_key is None:
        raise Exception(f"Environment variable {env_var_name} is not set")
    print(f"API Key read from environment variable {env_var_name}: {api_key}")  # Debugging line
    return api_key
#Load the API key
OPENAI_API_KEY = load_api_key('OPENAI_API_KEY')
FLASK_SECRET_KEY = load_api_key('FLASK_SECRET_KEY')
OPEN_WEATHER_MAP_API_KEY = load_api_key('OPEN_WEATHER_MAP_API_KEY')
GOOGLE_MAPS_API_KEY = load_api_key('GOOGLE_MAPS_API_KEY')

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
# Shared data structure
data = {
    "telemetry": None
}

# Connect to the drone's telemetry feed
#vehicle = connect('udpin:192.168.0.144:14550', wait_ready=False) # wait_ready=False is important to avoid timeout error
# to bypass the connection to the connection to the drone's telemetry feed, and use only the user input address, comment out the above line and uncomment the below line


#vehicle = connect('udpin:192.168.188.148:14550', wait_ready=False) # if using mobile phone TRY use these settings , mobile hotspot internet may not work
# if using mobile phone, change Rosetta app host 1 setting to : 192.168.188.148
#vehicle.initialize(8,30)
#vehicle.wait_ready('autopilot_version')

#connect to drone gps coordinates to get location
#print("connect to the drone's telmetry data using dronekit") 

#def fetch_telemetry():
#    while True:
#        # Fetching telemetry data from the drone
#        lat = vehicle.location.global_frame.lat
#        lon = vehicle.location.global_frame.lon
        
        # Check if lat and lon are valid before updating data["telemetry"]
#        if lat is not None and lon is not None:
#            data["telemetry"] = {"lat": lat, "lon": lon}
#            print(f"Fetched telemetry: {data['telemetry']}")  # Add logging
#        else:
#            print("Invalid telemetry data fetched")  # Add logging for invalid data
        
#        time.sleep(10)  # Update every 10 seconds




@app.route("/")
def index():
    return render_template("index.html")

@app.route("/telemetry")
def get_telemetry():
    try:
        telemetry = data.get('telemetry')
        if telemetry and 'lat' in telemetry and 'lon' in telemetry:
            print(f"Sending telemetry: {telemetry}")  # Add logging
            return jsonify(telemetry)
        else:
            raise ValueError("Invalid telemetry data")
    except Exception as e:
        print(f"Error getting telemetry: {str(e)}")  # Log error message
        return jsonify(error=str(e)), 500  # Send error message to frontend



# Define a default address
DEFAULT_ADDRESS = "5057 Edgewater Court, Savage, MN 55378"

# Initialize the geolocator
geolocator = Nominatim(user_agent="My_Drone_WeatherWatch")

def geocode_address(address):
    sleep(1)  # Delay to respect rate limit
    location = geolocator.geocode(address)
    if location:
        if 'address' in location.raw:
            # Get the address dictionary
            address_dict = location.raw['address']
            # Try to get the city from the 'town', 'city', or 'village' key
            city = address_dict.get('town') or address_dict.get('city') or address_dict.get('village')
            print("City found in 'address' field.")
        else:
            # Split the display_name into a list
            display_name = location.raw['display_name'].split(', ')
            # The city is usually the fourth to last item in the list
            city = display_name[-5] if len(display_name) > 5 else None
            print("'address' field not found. Using 'display_name' field for geocoding.")
        print(f"Geocoded Address: {address} -> Lat: {location.latitude}, Lon: {location.longitude}, City: {city}")
        print(f"Display Name: {location.raw['display_name']}")
        return location.latitude, location.longitude, city
    else:
        print("Could not geocode address")
        return None, None, None
    

@app.route('/geocode-address', methods=['POST'])
def geocode_address_route():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data received or bad JSON'}), 400

    address = data.get('address', DEFAULT_ADDRESS)
    lat, lon = geocode_address(address)  # Use the standalone function

    if lat and lon:
        return jsonify({'latitude': lat, 'longitude': lon})
    else:
        return jsonify({'error': 'Address not found or other error'}), 400

    
@app.route('/weather', methods=['POST'])
def weather():
    # Retrieve form data
    city = request.form.get('city', '')  # Default to empty string if not provided
    state_code = request.form.get('state_code', '')  # Default to empty string if not provided
    address = request.form.get('address', '')  # Default to empty string if not provided
    gps_source = request.form.get('gpsSource', 'address')  # Default to 'address' if not provided
    api_key = OPEN_WEATHER_MAP_API_KEY # replace with your OpenWeatherMap API key

    # Default to Minneapolis if all inputs are blank or improperly formatted
    if not city and not state_code and not address:
        flash(" TRY AGAIN - All field inputs are blank OR the address is incorrectly formatted. Please enter a city, state code, or address.")
        return redirect(url_for('index'))  # Replace 'input_form' with the name of your input form route
    else:
        # Decide the source of GPS coordinates
        if gps_source == 'address' and address:
            # Geocode the address to get the coordinates
            lat, lon, city = geocode_address(address)
            if lat is None or lon is None:  # Geocoding failed or returned no result
                city = "Minneapolis"  # Default to Minneapolis
                lat, lon, _ = geocode_address(city)  # Geocode the default city
        else:
            lat, lon = None, None

    # Fetch weather data
    if lat is not None and lon is not None:
        # Fetch weather data for the geocoded coordinates using One Call API
        weather_data = get_weather(city=city, lat=lat, lon=lon, api_key=api_key)
        weather = print_weather(weather_data, city=city, api_type='onecall')
        print(city) # this is the city that the user inputs
    else:
        # Fetch weather data using city and state code using Standard API
        weather_data = get_weather(city=city, state_code=state_code, api_key=api_key)
        print(weather_data) # this is the weather data when the user inputs the city and state
        weather = print_weather(weather_data, api_type='standard')

    if weather_data:
        # Fetch NWS alerts for the state
        alerts_data = fetch_nws_alerts(state_code)
        # Pass both weather and alerts data to the frontend
        return render_template('weather.html', weather=weather, alerts=alerts_data, google_maps_key=GOOGLE_MAPS_API_KEY)
    else:
        return jsonify({'error': 'Failed to fetch weather data'}), 500



def get_weather(city=None, state_code=None, lat=None, lon=None, api_key=None):
    if lat and lon:
        # Use One Call API for coordinates
        base_url = "https://api.openweathermap.org/data/3.0/onecall"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'imperial'
        }
        print(f"Requesting weather data from: {base_url} with params: {params}") # Add logging
    elif city:
        # Use standard API for city/state
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        query = f"{city},{state_code},US" if state_code else f"{city},US"  # Handle case where state_code might be empty
        params = {
            'q': query,
            'appid': api_key,
            'units': 'imperial'
        }
    else:
        # Handle case where neither coordinates nor city/state are provided
        return None
    
    print(f"Requesting weather data from: {base_url} with params: {params}")

    response = requests.get(base_url, params=params)
    #print(f"Received response: {response.json()}")  # Add logging
    return response.json()

def degrees_to_cardinal(d):
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    ix = round(d / (360. / len(dirs)))
    return dirs[ix % len(dirs)]

def print_weather(data, city=None, api_type='standard'):
    # Initialize a dictionary to hold the weather data
    weather = {}

    # Handle the response from the One Call API
    if api_type == 'onecall':
        current_weather = data.get('current', {})
        weather_description = current_weather.get('weather', [{}])[0].get('description', 'N/A')
        weather_icon_code = current_weather.get('weather', [{}])[0].get('icon', 'N/A')
        weather_icon_url = f"http://openweathermap.org/img/wn/{weather_icon_code}.png"
        weather = {
            'city': city,
            'temperature': current_weather.get('temp', 'N/A'),
            'humidity': current_weather.get('humidity', 'N/A'),
            'pressure': round(current_weather.get('pressure', 0) * 0.0295299830714, 2) if current_weather.get('pressure') else 'N/A',
            'wind_speed': current_weather.get('wind_speed', 'N/A'),
            'wind_direction': degrees_to_cardinal(current_weather.get('wind_deg', 0)) if current_weather.get('wind_deg') else 'N',
            'timestamp': datetime.fromtimestamp(current_weather.get('dt', 0)),
            'latitude': data.get('lat', 'N/A'),
            'longitude': data.get('lon', 'N/A'),
            'weather_description': weather_description,
            'weather_icon': weather_icon_url,
        }

    # Handle the response from the standard weather API
    elif api_type == 'standard':
        main = data.get('main', {})
        wind = data.get('wind', {})
        weather = {
            'city': data.get('name', 'Unknown'),
            'temperature': main.get('temp', 'N/A'),
            'humidity': main.get('humidity', 'N/A'),
            'pressure': round(main.get('pressure', 0) * 0.0295299830714, 2) if main.get('pressure') else 'N/A',
            'wind_speed': wind.get('speed', 'N/A'),
            'wind_direction': degrees_to_cardinal(wind.get('deg', 0)) if wind.get('deg') else 'N',
            'timestamp': datetime.fromtimestamp(data.get('dt', 0)),
            'latitude': data.get('coord', {}).get('lat', 'N/A'),
            'longitude': data.get('coord', {}).get('lon', 'N/A'),
        }

    # Return the formatted weather data
    return weather

@app.route("/nws-alerts")
def get_nws_alerts():
    state_code = request.args.get('state')  # Get the state code from the query parameters
    alerts = fetch_nws_alerts(state_code)
    if alerts:
        return jsonify(alerts)
    else:
        return jsonify(error="Failed to fetch NWS alerts"), 500

def fetch_nws_alerts(state_code=None):
    endpoint = "https://api.weather.gov/alerts/active"
    headers = {
        "User-Agent": "Weather_APP (bestevez100@gmail.com)",
        "Accept": "application/geo+json"
    }
    response = requests.get(endpoint, headers=headers)
    if response.status_code == 200:
        alerts = response.json()
        processed_alerts = []

        for alert in alerts['features']:
            if alert['properties']['areaDesc'].endswith(state_code):
                if alert['geometry']:
                    processed_alerts.append(alert)
                else:
                    # Fetch additional data for alerts with null geometry
                    for zone_url in alert['properties']['affectedZones']:
                        zone_data = requests.get(zone_url, headers=headers).json()
                        if 'geometry' in zone_data and zone_data['geometry']:
                            alert['geometry'] = zone_data['geometry']
                            processed_alerts.append(alert)
                            break  # Assuming one valid zone data is enough

        return {'features': processed_alerts}
    else:
        print(f"Error fetching NWS alerts: {response.status_code}")
        return None
    
# Initialize the OpenAI client with your API key
client = openai.OpenAI(api_key=OPENAI_API_KEY)

@app.route("/send-chat", methods=["POST"])
def send_chat():
    data = request.get_json()
    user_message = data['message']

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": user_message}
            ],
            model="gpt-3.5-turbo",
        )
        # Accessing the response content correctly
        response_content = chat_completion.choices[0].message.content
        return jsonify({'response': response_content})
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")  # Log the error for debugging
        return jsonify({'error': str(e)}), 500

#if __name__ == "__main__":
    # Start the background thread
    # Start the Flask app
    #threading.Thread(target=fetch_telemetry, daemon=True).start()
    # app.run(host='0.0.0.0', port=5800, debug=False)
