import sys
import os
import json
import openai
import time
from dotenv import load_dotenv
from logging_functions import log_message

# Load variables from .env file
load_dotenv()

# Access environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
model = "gpt-3.5-turbo"

#Retriece the JSON string and return the JSON object. Useful for ensuring JSON is valid from GPT-3.5 responses
def retrieve_JSON(json_string):

    start = json_string.find("{")
    end = json_string.rfind("}")
    split = json_string[start:end+1]

    try:
        temp_json = json.loads(split)
    except Exception as e:
        print("Error: " + str(e) + "\n\n" "JSON string:" + json_string)
        sys.exit(1)
    
    return temp_json

#Get the GPT-3.5 response and return the content
def get_gpt_response_content(role, prompt):
    
        try:
            response = openai.ChatCompletion.create(
                    model=model,
                    temperature=0.4,
                    messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": prompt},
                ]
            )
        except openai.error as e:
            print("Error: " + str(e))
            print("Waiting 1 second to try again...")
            time.sleep(1)
            return get_gpt_response_content(role, prompt)
        
        log_message("\n\nGPT-3.5 Response:\n" + "Role: " + role + "\n" + "Prompt: " + prompt + "\n" + "Response: " + response.choices[0].message.content + "\n\n")

        return ''.join(response.choices[0].message.content)