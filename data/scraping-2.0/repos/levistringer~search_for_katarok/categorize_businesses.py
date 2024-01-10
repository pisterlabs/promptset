import time
import utils
import os
import openai
# import tiktoken
import json
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

'''
business_names = # List of business json objects
    business = 
        {   'name': 'business name',
            'address: 'business address',
            'phone': 'business phone number',
        }
'''


def open_JSON_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=2000):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]


def count_average_number_of_characters_per_object(object_list):
    total_characters = 0
    for object in object_list:
        total_characters += len(str(object))
    return total_characters / len(object_list)


def get_businesses(business_names_filename):
    return open_JSON_file(business_names_filename)


def get_categories(businesses):
    list_of_businesses = [business['name']for business in businesses]

    delimiter = '####'  # Nice because it's treated as one token
    system_message = f"""
    Do not output any additional text that is not in JSON format.
    You will be provided with a python list of business names. \ 
    The business names will be delimited with {delimiter} characters.

    Output a python list of json objects, where each object has the following format:
        "category": <one of Consumer Products, Manufacturing Industries, Service Industries, Technologies, Non-profit>,\
    AND
       "industry": <A single industry that must be predicted> 
    AND "name": <the name of the business>
    
    Categorize each business into the predicted industry. \
    """
    # 'industry': <a single industry that must be found in the allowed industries below>
    # Allowed industries: {industries_and_category}
    messages = [
        {'role': 'system',
         'content': system_message},
        {'role': 'user',
         'content': f"{delimiter}{list_of_businesses}{delimiter}"},
    ]
    completion = get_completion_from_messages(messages)
    return completion


def main():
    open_file_path = 'belleville_businesses.json'
    processed_file_path = 'belleville_businesses_categorized.json'

    businesses = get_businesses(open_file_path)
    print(f"Total number of businesses: {len(businesses)}")

    average_char_per_obj = count_average_number_of_characters_per_object(
        businesses)

    print(
        f"""On average {int(average_char_per_obj)} characters per business object""")

    average_char_per_token = 4
    businesses_per_request = int(
        2000 / (average_char_per_obj / average_char_per_token))

    print(
        f"""With average of {int(average_char_per_obj / 4) } tokens per object going to try {businesses_per_request} businesses per request""")
    categorized_businesses = []
    for i in range(0, len(businesses), businesses_per_request):
        print(f"Processing businesses {i} to {i + businesses_per_request}")

        categorized_businesses.append(get_categories(
            businesses[i:i + businesses_per_request]))
        time.sleep(45)

    with open(processed_file_path, 'w') as fp:
        for item in categorized_businesses:
            # write each item on a new line
            fp.write(item)
    print('Done!')

    # industries_and_categories = utils.get_industries_and_categories()
    # get_categories('belleville_businesses.json', industries_and_categories)
main()
