#import subprocess
import os
# import openai
from dotenv import load_dotenv
import requests
import json

load_dotenv()#load the env vars

def get_gpt_response(message):
    print('in gpt' + message)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise Exception("OPENAI_API_KEY environment variable not found")
    url = "https://api.openai.com/v1/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    prompt = f"User: {message}"
    data = {
        "model": "text-davinci-003",
        "prompt": prompt,
        "max_tokens": 100,
        "n": 1, 
        #"stop" value to ["\nAssistant:"] to ensure that the model stops generating text at the beginning of the next assistant's response.
        "stop": ["\nAssistant:"], 
        "temperature": 0.7
        }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(f"Response status code: {response.status_code}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Response content: {response_data}")
        return response.json()["choices"][0]["text"].strip()
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")
    
#     # Test the function
# try:
#     result = get_gpt_response("What is the capital of France?")
#     print("GPT response:", result)
# except Exception as e:
#     print(e)





