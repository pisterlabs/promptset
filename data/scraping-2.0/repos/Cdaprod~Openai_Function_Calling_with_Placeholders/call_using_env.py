import os
from dotenv import load_dotenv
import requests
import json
import openai

# Load environment variables
load_dotenv()

# Define headers
headers = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {os.getenv('OPENAI_API_KEY')}",
}

# Define data for the POST request
data = {
    "model": os.getenv('MODEL_NAME'),
    "messages": [
        {"role": os.getenv('SYSTEM'), "content": os.getenv('SYS_CONTENT')},
        {"role": os.getenv('USER'), "content": os.getenv('USER_CONTENT')}
    ],
    "functions": [
        {
            "name": os.getenv('FUNCTION_NAME'),
            "description": os.getenv('FUNCTION_DESCRIPTION'),
            "parameters": {
                "type": os.getenv('PARAMETER_TYPE'),
                "properties": {
                    os.getenv('PROPERTY_NAME_1'): {
                        "type": os.getenv('PROPERTY_TYPE_1'),
                        "description": os.getenv('PROPERTY_DESCRIPTION_1')
                    },
                    os.getenv('PROPERTY_NAME_2'): {
                        "type": os.getenv('PROPERTY_TYPE_2'),
                        "description": os.getenv('PROPERTY_DESCRIPTION_2')
                    }
                },
                "required": [os.getenv('REQUIRED')]
            }
        }
    ]
}

# Make the POST request
response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, data=json.dumps(data))

# Print the response
print(response.json())
