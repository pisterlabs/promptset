from datetime import datetime
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_KEY")
# https://github.com/openai/openai-python/discussions/742
client = OpenAI(api_key=api_key)

FIRST_PROMPT = (
    "You are a mindful bot that helps the user to get a lot of rest. " +
    "You will ask the user when did they rest the last time, and the user will respond with when they rested last time, and what activity they did. " +
    "If the user hasn't taken a break for a few hours, you would suggest the user to take a break. " +
    "You will choose one and only of the following options: stretch, rest, nap, get a glass of water, meditate, talk to a friend, pet your cat or dog, go for a walk, or anything you can think of. " +
    "If the user hasn't taken a break for more than 24 hours, you will tell the user to stop working immediately. " +
    "Keep your response very short, just a sentence or two. " +
    "And don't ask follow-up questions. Do not ask user if there's anything else you can help with. Just end the conversation."
)

JSON_PROMPT = (
    "You are a data processing bot that will organize the user's rest data. " +
    "The user will tell you when they last took a rest. " +
    "Look for the following information about the user's rest: 'activity_name', 'activity_duration', and 'activity_time'." +
    "You will return these three pieces of information as a JSON object. " +
    "Don't ask any follow-up question, just do your best to interpret the user's response into the three pieces of information. " +
    "If you don't know what the activity_duration is, make it 10 minutes by default. " +
    "For activity_time, return a string like '2021-10-10 10:10:10'. And also you know the current time is " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
    "For activity_name, if you don't know what the activity is, return 'Unknown rest activity'. "
)

def json_answer(last_rest, model="gpt-3.5-turbo", max_tokens=800, stop_sequence=None):
    messages = [
         {"role": "system", "content": JSON_PROMPT},
         {"role": "assistant", "content": "When did you last take a rest?"},
         {"role": "user", "content": last_rest},
    ]

    try:
        # Create a completions using the question and context
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""


def first_answer(last_rest, model="gpt-3.5-turbo", max_tokens=800, stop_sequence=None):
    messages = [
         {"role": "system", "content": FIRST_PROMPT},
         {"role": "assistant", "content": "When did you last take a rest?"},
         {"role": "user", "content": last_rest},
    ]

    try:
        # Create a completions using the question and context
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""


