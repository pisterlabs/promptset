import os
import openai
from dotenv import load_dotenv
from pdf_parser import parse_pdf
import json
import pathlib
from datetime import datetime, timedelta, date

import requests
url = 'https://cloud.mindsdb.com/api/sql/query'
cookies = {'session': '273trgsehgrui3i2riurwehe'}

num_days = 0

def load_config():
    """Load and return configuration variables from .env file."""
    load_dotenv('/home/ubuntu/App/.env')
    openai_api_key = os.getenv("OPENAI_API_KEY")
    completion_model = "gpt-4-0613"
    return openai_api_key, completion_model

def get_completion(messages, openai_api_key, completion_model):
    """Create and return a GPT-4 completion."""
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model=completion_model,
        messages=messages,
        temperature=0,
        max_tokens=6000,
    )
    return response

def count_days(data):
    day_keys = [key for key in data.keys() if key.startswith("Day")]
    num_days = len(day_keys)
    return num_days

def pretty_print_json(data):
    pretty_data = json.dumps(data, indent=4)
    pretty_data = json.loads(pretty_data)
    num_days = count_days(data)
    for i in range(1, num_days+1):
        day = f"Day{i+1}"
        if day in pretty_data:  # Check if the day exists in the data
            a = str(date.today() + timedelta(days=i+1))
            q1 = f"""INSERT INTO my_calendar3.events(start_time, end_time, summary, description) VALUES ({a + " 08:00:00"},{a + " 08:30:00"}, 'Breakfast', {pretty_data[day]["Breakfast"]});"""
            q2 = f"""INSERT INTO my_calendar3.events(start_time, end_time, summary, description) VALUES ({a+ " 12:00:00"}, {a + " 12:30:00"}, 'Lunch', {pretty_data[day]["Lunch"]});"""
            q3 = f"""INSERT INTO my_calendar3.events(start_time, end_time, summary, description) VALUES ({a + " 18:00:00"}, {a + " 18:30:00"}, 'Dinner', {pretty_data[day]["Dinner"]});"""
            q4 = f"""INSERT INTO my_calendar3.events(start_time, end_time, summary, description) VALUES ({a + " 10:00:00"}, {a + " 10:30:00"}, 'Morning Workout', {pretty_data[day]["Workout1"]});"""
            q5 = f"""INSERT INTO my_calendar3.events(start_time, end_time, summary, description) VALUES ({a + " 15:00:00"}, {a + " 15:30:00"}, 'Evening Workout', {pretty_data[day]["Workout2"]});"""
            print(q1)
            resp1 = requests.post(url, json={'query': q1}, cookies=cookies)
            resp2 = requests.post(url, json={'query': q2}, cookies=cookies)
            resp3 = requests.post(url, json={'query': q3}, cookies=cookies)
            resp4 = requests.post(url, json={'query': q4}, cookies=cookies)
            resp5 = requests.post(url, json={'query': q5}, cookies=cookies)
        else:
            print('No plan available for ' + day)


def get_meal_suggestions(query, bloodwork, openai_api_key, completion_model):
    """Obtain meal suggestions from GPT-4 based on the user input."""
    if bloodwork is None:
        user_prompt = f"Generate a n-day meal and workout plan that revolves around the user's needs as specified in their {query}. Formatted as JSON and the JSON is never between triple tick marks, single quotes, double quotes, etc. Return JSON obect only, no greetings or comments or introductory or conlcuding remarks. Ensure all keys and values are between pair of double quotes."
    else:
        user_prompt = f"Generate a n-day meal and workout plan that revolves around the user's needs as specified in their {query} and their bloodwork as per their results: {bloodwork}. Formatted as JSON and the JSON is never between triple tick marks, single quotes, double quotes, etc. Return JSON obect only, no greetings or comments or introductory or conlcuding remarks."
    messages = [
        {"role": "system", "content": "You are a helpful assistant that suggests detailed meal plans and workout plans. When asked, provide clear meal suggestions for a specified number of days, along with a paragraph containing their ingredients and nutritional facts including total calories, total carbohydrates, dietary fiber, sugar, protein, vitamins (A, C, E, B6), calcium, iron, total fat, and saturated fat. Also provide a workout plan for each day. Format your response in a structured, JSON-friendly manner that is easy to parse and short. There should be three meals and two workout plans per day. The response should have five keys for each day: breakfast, lunch, dinner, workout1, and workout2. Ensure all keys and values are between pair of double quotes."},
        {"role": "user", "content": user_prompt},
    ]
    response = get_completion(messages, openai_api_key, completion_model)
    print(response)
    return json.loads(response.choices[0]["message"]["content"])
    
def final_answer(query, bloodwork, meal_suggestions, openai_api_key, completion_model):
    """Obtain meal suggestions from GPT-4 based on the user input."""
    if bloodwork is None:
        user_prompt = f"Generate a cogent answer where a user's n-day meal and workout plan that revolves around their user's needs as specified in their {query}, and answered in these {meal_suggestions}."
    else:
        user_prompt = f"Generate a cogent answer where a user's n-day meal and workout plan that revolves around their user's needs as specified in their {query}, by their blood work as per their results: {bloodwork}, and answered in these {meal_suggestions}."
    messages = [
        {"role": "system", "content": "You have been a helpful assistant that suggested detailed meal plans and workout plans. When asked, provide a cogent and authorotative answer that explains to the user how their plans fit their needs."},
        {"role": "user", "content": user_prompt},
    ]
    response = get_completion(messages, openai_api_key, completion_model)
    return response.choices[0]["message"]["content"]



def start_chat(openai_api_key, completion_model, query, bloodwork_path=None):
    """Start the chat."""
    if bloodwork_path is not None:
        if isinstance(bloodwork_path, str) and pathlib.Path(bloodwork_path).exists():
            # If the provided bloodwork_path is a path to an existing file, parse it as a PDF
            bloodwork = parse_pdf(bloodwork_path)
        else:
            # If it's not a path to an existing file, treat it as the bloodwork results directly
            bloodwork = bloodwork_path
    else:
        bloodwork = None
    meal_suggestions = get_meal_suggestions(query, bloodwork, openai_api_key, completion_model)
    pretty_print_json(meal_suggestions)
    final_response = final_answer(query, bloodwork, meal_suggestions, openai_api_key, completion_model)
    return meal_suggestions, final_response


if __name__ == "__main__":
    openai_api_key, completion_model = load_config()
    # Start the chat
    start_chat(openai_api_key, completion_model)

