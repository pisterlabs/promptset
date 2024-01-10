''' An application to generate a recipe and an image from a list of ingredients.
    THis uses calls to open API LLMs for recipie and image of the dish.'''

import openai
from dotenv import load_dotenv
import os

import requests
import shutil
import re

# load environment variables from .env file
load_dotenv()

# get api key from environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]

# Create the prompt for the recipe
def create_recipe_prompt(ingredients):
    list_of_ingredients = '\n    #      '.join(ingredients)
    prompt = f"""
    # Create a detailed recipe based only on the following listed ingredients.
    # Additionally, assign a title starting with 'Recipe Title' to the recipe.
    #
    # Recipe Title:
    #
    # Ingredients:
    #      {list_of_ingredients}
    #
    # Instructions:"""
    return prompt

# A function to return the recipie title from the prompt result using regex
def extract_title(prompt_result):
    title = re.findall(r'^.*Recipe Title: .*$', prompt_result, re.MULTILINE)
    return title[0].replace('Recipe Title: ', '')

# Create the prompt for the image based of passed recipe title
def create_image_prompt(recipe_title):
    prompt = f"""
    # Create a photo-realistic image of a meal of {recipe_title} on a table .
    # Do not include the title.
    #  """
    return prompt

def create_image_prompt2(recipe_title):
    prompt = f"""{recipe_title}, professional food photography, 15mm, studio lighting, 1/125s, f/5.6, ISO 100, 5500K, 1/4
    # Do not include the title.
    #  """
    return prompt

# Download and save the image returned from DALLE
def save_image(image_response, filename):
    image_url = image_response['data'][0]['url']

    image_res = requests.get(image_url, stream=True)
    if image_res.status_code == 200:
        with open(filename, 'wb') as image_file:
            shutil.copyfileobj(image_res.raw, image_file)
    else:
        print('Image couldn\'t be retreived')

    return image_res.status_code

prompt = create_recipe_prompt(['chicken', 'rice', 'broccoli'])

response = openai.Completion.create(engine="text-davinci-003",
                                    prompt=prompt,
                                    max_tokens=256,
                                    temperature=0.8)

result_text =response['choices'][0]['text']
#print(result_text)

title = extract_title(result_text)
print(title)

# Create the image using DALL-E
image_prompt = create_image_prompt2(title)
image_response = openai.Image.create(prompt=image_prompt,
                                    n=1,
                                    size='1024x1024')

result = save_image(image_response, 'dev/recipe_image.jpg')
print(f'Image saved: {result}')