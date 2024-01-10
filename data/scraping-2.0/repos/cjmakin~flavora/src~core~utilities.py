import os
import hashlib
import openai
import requests
from dotenv import load_dotenv
from pathlib import Path
from django.conf import settings

load_dotenv()

openai.api_key = os.environ['_OPENAI_API_KEY']
BASE_DIR = Path().resolve()

system_prompt = """You are an expert chef. You will be given a list of ingredients, a desired maximum cooking time, and dietary preferences.
You will then generate a recipe for a dish that contains those ingredients, is within the cooking time, and meets the dietary preferences.
You will output a string with the following information exactly in this order: name, description, instructions, ingredients, cooking_time. 
Each piece of information will be separated by $$$.
The instructions and ingredients will be newline-separated strings.
Assume the user has basic cooking knowledge. Include ingredient quantities.
Basic Example Output: [name]$$$[description]$$$[instructions]$$$[ingredients]$$$[cooking time]
"""

user_prompt = """
Ingredients: {}
Dietary preferences: {}
Maximum cooking time: {}"""

example_prompt = """
Ingredients: shrimp, garlic, olive oil, cumin, cayenne pepper
Dietary preferences: None
Maximum cooking time: 2 hours"""

example_response = "Spicy Grilled Shrimp Skewer$$$These grilled shrimp skewers are perfect for a summer barbecue or a quick weeknight dinner. The spicy marinade adds a kick of flavor to the succulent shrimp$$$", \
    "1. Soak wooden skewers in water for 30 minutes to prevent them from burning on the grill\n", \
    "2. In a small bowl, whisk together olive oil, minced garlic, smoked paprika, cumin, cayenne pepper, salt, and black pepper.\n", \
    "3. Thread the shrimp onto the skewers, about 4-5 per skewer.\n", \
    "4. Brush the shrimp with the marinade, making sure to coat all sides.\n", \
    "5. Preheat the grill to medium-high heat.\n", \
    "6. Place the shrimp skewers on the grill and cook for 2-3 minutes per side, or until the shrimp are pink and opaque.\n", \
    "7. Serve hot with a squeeze of lemon juice and chopped cilantro.$$$1 pound of large shrimp, peeled and deveined$$$", \
    "2 tablespoons of olive oil\n", \
    "2 cloves of garlic, minced\n", \
    "1 teaspoon of smoked paprika\n", \
    "1 teaspoon of cumin\n", \
    "1/2 teaspoon of cayenne pepper\n", \
    "1/2 teaspoon of salt\n", \
    "1/4 teaspoon of black pepper\n", \
    "Wooden skewers\n", \
    "Lemon wedges and chopped cilantro for serving$$$20 minutes"


def decompose_response(response):
    response_arr = response.strip().split("$$$")

    response_dict = {
        'name': response_arr[0].strip(),
        'description': response_arr[1].strip(),
        'instructions': response_arr[2].strip(),
        'ingredients': response_arr[3].strip(),
        'cooking_time': response_arr[4].strip()
    }
    return response_dict


def generate_recipe(user_email, ingredients, cooking_time, food_preferences):
    hashed_email = hashlib.sha256(user_email.encode('utf-8')).hexdigest()

    ingredients = [ingredient for ingredient in ingredients if ingredient]
    cooking_time = '5 hours' if cooking_time == '' else cooking_time
    food_preferences = 'None' if food_preferences == [] else food_preferences

    print(f'Attempting to generate recipe for {user_email}')

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":  example_prompt},
                {"role": "assistant", "content": ''.join(example_response)},
                {"role": "user", "content": user_prompt.format(
                    ingredients, food_preferences, cooking_time)}
            ],
            n=1,
            temperature=1,
            user=hashed_email,
        )
    except Exception as e:
        print(e)
        return -1

    try:
        recipe = decompose_response(response.choices[0].message.content)
    except Exception as e:
        print(e)
        return -1

    return recipe


def generate_image(recipe_description, email):
    hashed_username = hashlib.sha256(email.encode('utf-8')).hexdigest()

    print(f'Attempting to generate image for {email}')

    # Request arguments
    prompt = f'Create an image with no text of: {recipe_description}'
    size = '512x512'
    n = 1
    response_format = 'url'

    try:
        response = openai.Image.create(
            prompt=prompt, n=n, size=size, user=hashed_username, response_format=response_format)
        img_url = response['data'][0]['url']

        return img_url

    except Exception as e:
        print(e)
        return -1


def save_image(img_url, user_id, recipe_id):

    print(f'Attempting to save image for user {user_id}')

    filename = f'{recipe_id}.png'
    filepath = os.path.join(settings.MEDIA_ROOT, 'recipes', str(user_id))

    try:
        os.makedirs(filepath, exist_ok=True)

        with open(f'{filepath}/{filename}', 'wb') as f:
            f.write(requests.get(img_url).content)

        return f'{filepath}/{filename}'

    except Exception as e:
        print(e)
        return -1
