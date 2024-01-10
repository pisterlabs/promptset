import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GPT4_APIKEY")
BASE_PROMPT = """
                The following data includes what a user consumed for breakfast, lunch, dinner, and snacks and desserts,
                with respective satiety ratings for each meal. A satiety rating of 1 signifies that the user was “Very Hungry”
                after consuming the meal, a level of 3 signifies “Just Right”, and a level of 5 signifies “Very Full”.
                Breakfast: {}, satiety rating {}
                Lunch: {}, satiety rating {}
                Dinner: {}, satiety rating {}
                Snacks: {}, satiety rating {}

                You are going to write a response that will be featured in a health app. Taking into account what the user has eaten so far in the week and the nutrient information of those foods(protein, carbs, fat, etc),
                give them a one sentence dietary tip for the day. For example, if the user reported eating a lot of desserts at a family gathering,
                tell them "Since you have already consumed a high number of carbohydrates this week, try to avoid sugary splurges for the next few days."
                Essentially, point out a poor dietary choice they made in the day, if they made one. In addition, take the following user's health data
                into account when writing your response:
                Age: {}, Sex: {}, Height: {}, Weight: {}, and Ethnicity: {}
              """

def generate_prompt(arguments):
    return BASE_PROMPT.format(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7],
                              arguments[8], arguments[9], arguments[10], arguments[11], arguments[12])

def call_gpt4(arguments):
    openai.api_key = API_KEY

    GPTprompt = generate_prompt(arguments)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": GPTprompt
            },
        ],
        temperature=0.7
    )
    
    contentStr = response.choices[0].message.content
    return contentStr