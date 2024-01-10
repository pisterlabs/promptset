import os
from dotenv import load_dotenv
import openai

# Load the .env file to get the OPENAI_API_KEY
load_dotenv()

# Retrieve the API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = openai_api_key

    
import requests

# Replace the URL with the appropriate URL of your local OpenAI API server
url = "http://localhost:5001/v1/engines"

# Make a GET request to the server
response = requests.get(url)
print(response)

# Parse the JSON response
data = response.json()

# Extract and print the model name or ID
print("Model Information:", data)