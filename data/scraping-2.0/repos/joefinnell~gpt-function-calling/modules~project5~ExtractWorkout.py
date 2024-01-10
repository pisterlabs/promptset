from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

client = OpenAI()
BASE_PROMPT = 'you create a name for it, and format it in the following JSON format: {"name":"Workout name", "exercises":[{"name":"Exercise name", "sets":5, "reps":5, "weights":225, "notes":"Heavy"}]}'
def create_workout_prompt():
    return 'Based on the workouts the user entered you create a workout that works on areas the user is missing. The workouts are in json form.' + BASE_PROMPT

def extract_workout_prompt():
    return 'You extract data about a workout, ' + BASE_PROMPT

def create_workout(content):
    return process_chat_completion(create_workout_prompt, content)

def extract_workout(content):
    return process_chat_completion(extract_workout_prompt, content)

def process_chat_completion(prompt_func, content):
    messages = [
        {"role": "system", "content": prompt_func()},
        {"role": "user", "content": content}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        messages=messages,
    )
    return json.loads(response.choices[0].message.content)