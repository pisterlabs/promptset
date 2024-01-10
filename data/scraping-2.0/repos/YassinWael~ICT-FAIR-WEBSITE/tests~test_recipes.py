import openai
from dotenv import load_dotenv
from os import getenv
from re import findall
from icecream import ic
# Loading API
load_dotenv('settings.env')
api = getenv('chatgpt')
client = openai.OpenAI(api_key=api)

pattern = "\*(.*?)\*"
pattern2 = "\*\*(.*?)\*\*" #chatgpt some times put two astericks

instructions_pattern = "\d\. (.*?)&"


user_ingredients = input("Tell us what you have, We only assume water and salt :  ") + " Water and salt and pepper"
country = input("Enter your country (Optional):  ")


def find_meal_names(text):
    meal_names = findall(pattern,text)
    if not meal_names:
        meal_names = findall(pattern2,text)
    return meal_names

def find_instructions(text,meal_names):
    dic = {}
    steps = findall(instructions_pattern,text)
    ic(steps)
    for meal in meal_names:
        for i in range(len(steps)):
            ic(i)
            ic(steps[meal_names.index(meal)-1:meal_names.index(meal)+4])
            dic[meal] = steps[meal_names.index(meal)-1:meal_names.index(meal)+4]
    ic(dic)
    return steps


def chatgpt(ingredients,country="mix of contries"):
    """
    Generates a chat-based prompt for the OpenAI GPT-3.5-turbo model to retrieve 5 food suggestions based on a given list of ingredients and an optional country.

    Args:
        api (str): The API key for accessing the OpenAI API.
        ingredients (str): The list of ingredients to base the food suggestions on.
        country (str, optional): The country to consider when generating the food suggestions. Defaults to "".

    Returns:
        None
    """
    
    response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    
    {"role": "system", "content": "As an Experienced chef, craft 3 meals from the specified country using only provided ingredients. Place meal names in asterisks, e.g., *French Toast*. Verify each step's ingredients; if missing, pick an alternative. Be creative, prioritize taste, If there is a popular meal with the ingredients given make sure to mention it. For the insturctions make sure to mention ONLY 5 STEPS AND MOST IMPORTANTLY END EVERY STEP WITH A &."},
    {"role": "user", "content": f"{ingredients}, from {country}"}
  ]
)
    
    chatgpt_answer = (response.choices[0].message.content)

    tokens_total = response.usage.total_tokens
    ic(tokens_total)

    print(chatgpt_answer)
  


    meals = find_meal_names(chatgpt_answer)
    print(meals)

    steps = find_instructions(chatgpt_answer,meals)
    ic(steps)
    return meals


chatgpt(user_ingredients,country)