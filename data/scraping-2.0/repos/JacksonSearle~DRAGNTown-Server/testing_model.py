import openai
import datetime
import os
import json
import time
from dotenv import load_dotenv

# from init_unreal import content_path
# This loads the API key as an environment variable
from config import path
print()
openai.api_key_path = (path + ".env")
load_dotenv(path)
# Maximum number of API calls per day
MAX_API_CALLS = 10000


def get_valid_response(prompt, keys, types):
    i = 0
    while True:
        if i != 0:
            print(f'Retry number {i}')
        # Get a response from the model
        response = query_model(prompt)

        # Try to extract a JSON object from the response
        start = response.find('{')
        end = response.rfind('}') + 1
        try:
            response_dict = json.loads(response[start:end])
        except json.JSONDecodeError:
            # If the extraction fails, try again
            continue

        # Check that all keys are present and have the correct type
        if all(key in response_dict and isinstance(response_dict[key], type_) for key, type_ in zip(keys, types)):
            return response_dict
        # If any key is missing or has the wrong type, try again
        i += 1

def query_model(prompt, _):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return model(messages)

def model(messages):
    # Load or initialize the count and date
    filename = path + 'prompts/api_calls.json'
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            call_count = data['text_count']
            last_call_date = datetime.datetime.strptime(data['date'], '%Y-%m-%d').date()
    else:
        call_count = 0
        last_call_date = datetime.date.today()

    # Check if it's a new day
    if datetime.date.today() != last_call_date:
        # If it's a new day, reset the count
        call_count = 0
        last_call_date = datetime.date.today()

    if call_count < MAX_API_CALLS:

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                messages=messages
            )
        except openai.error.ServiceUnavailableError as e:
            print('Server is busy, trying again in 3 seconds.')
            time.sleep(3)
            return model(messages)

        response_text = response['choices'][0]['message']['content']

        # Increment and save the count
        call_count += 1
        with open(filename, 'w') as f:
            json.dump({'image_count': data['image_count'], 'text_count': call_count, 'date': str(last_call_date)}, f)
        # print(f'API Call: {messages[1]["content"]}\n')
        return response_text

    else:
        print(f"API call limit of {MAX_API_CALLS} per day reached. Please wait until tomorrow.")


query_model("Who was the first president of the USA?", '')