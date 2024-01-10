import pytesseract
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import json
import openai

from flask import Flask, request, jsonify

app = Flask(__name__)

def recommend_menu(image_url, restriction):
    # Download the image from the URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Perform OCR
    text = pytesseract.image_to_string(img)

    # Your OpenAI API key
    api_key = openai_api

    # Your input prompt for GPT-3
    prompt = f"find all foods from the text: {text}"

    # Make a request to GPT-3
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.0,
        api_key=api_key
    )

    # Extract the generated ingredients list from the response
    ingredients = response.choices[0].text
    ingredients = ingredients[ingredients.find(':') + 1:].strip()

    # Your Edamam API credentials
    app_id = edamam_id
    app_key = edamam_key

    def find_ingredients(dish_name):
        # Make the API request
        url = f'https://api.edamam.com/api/food-database/parser'
        params = {
            'app_id': edamam_id,
            'app_key': edamam_key,
            'ingr': dish_name,
        }
        response = requests.get(url, params=params)
        data = response.json()
        # Initialize the index variable to None
        index_with_food_content_label = None
        # Iterate through the list
        for i, item in enumerate(data['hints']):
            if 'foodContentsLabel' in item['food']:
                index_with_food_content_label = i
                break
        ingredients_string = data['hints'][index_with_food_content_label]['food']['foodContentsLabel'].split(';')
        return ingredients_string

    dish_n_ingredients = []
    dishes = ingredients.strip().split(',')

    for dish in dishes:
        ingredients = find_ingredients(dish)
        dish_n_ingredients.append({'Dish': dish, 'Ingredients': ingredients})

    # Create a DataFrame from the list of dish-ingredient pairs
    df = pd.DataFrame(dish_n_ingredients)

    # Split the restriction string into a list
    restrictions = restriction.split(',')

    # Convert the restriction list to lowercase for case-insensitive comparison
    restrictions = [item.lower() for item in restrictions]

    # Filter the DataFrame to remove rows where any item in 'restriction' is in the list of ingredients
    df = df[~df['Ingredients'].apply(lambda ingredients: any(restriction in ingredient.lower() for ingredient in ingredients for restriction in restrictions))]

    # Reset the index after removing rows
    df.reset_index(drop=True, inplace=True)

    # Convert the 'Dish' column to a JSON-friendly list
    result = json.dumps(df['Dish'].tolist())

    return result
@app.route('/')
@app.route('/recommendation_menu', methods=['GET','POST'])
def recommendation_menu():
    # Get the image URL and restriction from the request data
    data = request.get_json()
    image_url = data['image_url']
    restriction = data['restriction']

    # Call the recommend_menu function
    result = recommend_menu(image_url, restriction)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
