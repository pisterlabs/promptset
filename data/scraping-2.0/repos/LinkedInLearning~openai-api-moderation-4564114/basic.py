# Baseline example of OpenAI's Moderation API.
# 
# - OpenAI Moderation guide: https://platform.openai.com/docs/guides/moderation
# - OpenAI Moderations API reference: https://platform.openai.com/docs/api-reference/moderations
# - OpenAI Python library: https://github.com/openai/openai-python

import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load the .env file
load_dotenv()

# Create a client
client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),
)

# Helper function to pretty print the response object
def print_response(response_obj):
    # Serialize the response object
    def serialize(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

    # Create a dictionary
    response_dict = response_obj.__dict__

    # Print the dictionary
    print(json.dumps(response_dict, indent=4, default=serialize))

# Create a chat prompt
prompt = "blond people are boring"

# Run prompt through the moderation API
mod = client.moderations.create(input=prompt)

# Print the full response object
print_response(mod)