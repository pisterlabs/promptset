from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import json

app = Flask(__name__)
CORS(app)  # CORS configuration

# Replace these values with your actual API keys
HF_API_TOKEN = "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"
CLAUDE_API_KEY = "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"

CHROMADB_PATH = './database/chroma'
METADATA_DB_PATH = './database/metadata.json'
TEMP_PDF_FILE_NAME = 'temp.pdf'
CHUNK_SIZE = 1000

embedder = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
)
chroma_client = Chroma(
    embedding_function=embedder,
    persist_directory=CHROMADB_PATH
)
claude = Anthropic(api_key=CLAUDE_API_KEY)

def claude_completion(prompt, model_name="claude-instant-1", max_tokens=500):
    location_prompt = f"{HUMAN_PROMPT} " + str(prompt) + f" {AI_PROMPT}"
    res = claude.completions.create(
        model=model_name,
        max_tokens_to_sample=max_tokens,
        prompt=location_prompt
    )
    return res.completion


def parse_location_from_response(response):
    try:
        # Try to parse the response as JSON
        response_json = json.loads(response)

        # Assuming the response is a JSON object, extract the location
        location = response_json.get('location', None)
        
        # Split the location string into latitude and longitude
        latitude, longitude = map(float, location.split(', '))
        
        # Return the latitude and longitude as a dictionary
        return {'latitude': latitude, 'longitude': longitude}
    except json.JSONDecodeError:
        # If parsing as JSON fails, treat the response as a plain string
        # Modify this part based on the actual structure of your response
        location = response.strip()  # Assuming location is a string in this case
        return location
    except Exception as e:
        print(f"Unexpected error in parse_location_from_response: {str(e)}")
        return None


def get_location_from_query(query):
    try:
        # Provide a prompt with a placeholder for the location in the answer
        prompt = """
        Context: User is asking about parking locations. First verify if the location is valid and if it is within singapore. If both are ture then, return only the latitude and longitude values alone. 
        Remember to answer only with the values of the latitude and longitude or leave it empty.

        Question: Where is the {location} parking area?
        Answer: {latitude}, {longitude}
        """.format(location=query, latitude=1.234567, longitude=2.345678)

        # Use Anthropic to complete the prompt
        response = claude_completion(prompt)

        # Parse the response to extract the location
        location = parse_location_from_response(response)

        return location
    except Exception as e:
        print(f"Error in get_location_from_query: {str(e)}")
        return None
    

def fetch_nearby_parking_locations(latitude, longitude, radius):
    try:
        apiUrl = f"http://127.0.0.1:3000/nearbyParking/residential_areas_nearby?latitude={latitude}&longitude={longitude}&radius={radius}"

        response = requests.get(apiUrl)

        if response.status_code == 200:
            data = response.json()
            return data.get("residentialAreas", [])
        else:
            print(f"Error fetching nearby parking locations. Status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching nearby parking locations: {str(e)}")
        return []


def fetch_parking_availability(location_id):
    try:
        apiUrl = f"http://192.168.68.101:3000/showOrUpdate/parking-availability?locationId={location_id}"

        response = requests.get(apiUrl)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error fetching parking availability. Status code: {response.status_code}")
            return {}
    except Exception as e:
        print(f"Error fetching parking availability: {str(e)}")
        return {}
    
def get_response_with_context(query, location, availability):
    # Construct a prompt with the provided information
    prompt = f"""
    Context: User is asking about parking locations and availability.
    Query: {query}
    Location: {location}
    Availability: {availability}
    Question: {query}
    Answer: """
    
    # Use Anthropic to complete the prompt
    response = claude_completion(prompt)
    
    return response


@app.route('/extract_location', methods=['POST'])
def extract_location():
    try:
        data = request.get_json()
        query = data.get('query')

        # Get location from query using Claude AI
        location_response = get_location_from_query(query)
        location = parse_location_from_response(location_response)

        print(location)
        
        if not location:
            return jsonify({'location': None, 'error': 'Invalid location or not within Singapore'})

        # Fetch nearby parking locations
        nearby_locations = fetch_nearby_parking_locations(location.split(',')[0], location.split(',')[1], radius=1000)

        if not nearby_locations:
            return jsonify({'location': location, 'error': 'No nearby parking locations found'})

        # Fetch availability for each nearby location
        availability_data = {}
        for loc in nearby_locations:
            location_id = loc.get('id')
            availability = fetch_parking_availability(location_id)
            availability_data[location_id] = availability

        # Get response from Claude AI with all the context
        response = get_response_with_context(query, nearby_locations, availability_data)

        return jsonify({'location': location, 'nearby_locations': nearby_locations, 'availability': availability_data, 'response': response})

    except Exception as e:
        print(f"Error in /extract_location: {str(e)}")
        return jsonify({'location': None, 'error': 'Internal server error'})


if __name__ == '__main__':
    app.run(debug=True)
