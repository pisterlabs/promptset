import os
import openai                             # pip install openai
OPENAI_API = os.environ.get('OPENAI_API')
openai.api_key = OPENAI_API


# This Data is given from databse or somethin idk
preferences = ["Chicken", "Pasta", "Spinach", "Pizza", "Burgers", "Tacos"]
filters  = ["Eco-friendly", "Meat", "Quick to Prepare"]


def generate_suggestion(preferences: list[str], filters: list[str]) -> str:

#Creating the Prompt
    pref = ""
    filt = ""


    for food in preferences:
        pref += food + ", "

    for item in filters:
        filt += item + ", "



    text_prompt = "Imagine you are hungry and need food ideas, give exactly 1 new meal idea(s) (something not in preferences) based on the preferences:" + pref + "with filters: "+ filt + "and only give the name of the meal"



    #print(text_prompt)


    #ChatGPT Response

    chatgpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text_prompt}],
        temperature=0.7,
        max_tokens=2000,
        top_p=0.95)

    #Get the list of foods
    response = chatgpt_response['choices'][0]['message']['content'].strip()
    #print(response)


    #Seperate into Meal and Ingredients

    # Split the response into meal and ingredients
    # meal, ingredients_str = response.split(":")
    # meal = meal.strip()  # Remove any leading/trailing whitespace

    # Split the ingredients into a list
    # ingredients = [ingredient.strip() for ingredient in ingredients_str.split(",")]

    # Now you have the variables meal and ingredients
    #print("Meal:", meal)
    #print("Ingredients:", ingredients)

    result = response

    return result