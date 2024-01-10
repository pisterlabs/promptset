import math
from flask import Flask, request, jsonify
import csv
from io import StringIO
from langchain.schema.messages import HumanMessage
from langchain.chat_models import ChatOpenAI
import base64
import pandas as pd
from constants import INGREDIENTS_DICT, INGREDIENTS_CSV_PATH

from dotenv import load_dotenv
load_dotenv(override=True)


app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/identify_ingredients', methods=['POST'])
def identify_ingredients():
    """
    Returns a list of ingredients present in the image, along with an emoji to be displayed in the app
    Input: picture
    Output: list of ingredients of the form [{'name': 'Orange', 'emoji': '🍊'}, ...]
    """

    # convert the picture to base64
    picture = request.files['picture']
    picture = picture.read()
    picture = base64.b64encode(picture)
    picture = picture.decode('utf-8')

    def ingredient_parser(img_base64, prompt):
        """Get the list of ingredients in the image."""
        chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)

        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
            ]
        )
        return msg.content

    # Prompt
    prompt = f"""# SnapChef
            Output an array containing a list of the unique ingredients in this picture, along with its corresponding emoji. For example, a valid output would be [{{'name': 'Orange', 'emoji': '🍊'}}, ...]. Your raw output will be parsed for JSON, so do NOT include backticks or any other data other than JSON.
            You are allowed to choose from the following list of ingredients: {INGREDIENTS_DICT}.
            """

    # Process the image with OpenAI's API
    response = ingredient_parser(picture, prompt)
    return response

@app.route('/find_recipes', methods=['POST'])
def find_highest_matching_recipe():
    """
    Returns a list of recipes that match the ingredients list, including the percent match for each recipe.
    Input: ingredients_list
    Output: list of recipes
    """
    print(f"Request: {request.get_json()}")

    # Convert CSV content to a dictionary
    ingredients_list = request.get_json()['ingredients_list']

    recipe_csv = pd.read_excel(INGREDIENTS_CSV_PATH)
    recipe_csv['percent_match'] = recipe_csv['ingredients'].apply(lambda x: (len(set(x.split(", ")).intersection(ingredients_list)) / len(x.split(", ")) * 100))
    recipe_csv['ingredients_have'] = recipe_csv['ingredients'].apply(lambda x: ingredients_to_dict(set(x.split(", ")).intersection(ingredients_list)))
    recipe_csv['ingredients_missing'] = recipe_csv['ingredients'].apply(lambda x: ingredients_to_dict(set(x.split(", ")).difference(ingredients_list)))
    recipe_csv['time'] = "30"
    recipe_csv['course'] = "Dinner"
    recipe_csv['servings'] = 4

    data = recipe_csv.sort_values(by=['percent_match'], ascending=False).head(3).to_json(orient='records')
    return data

def ingredients_to_dict(ingredients):
    """
    Converts a list of ingredients to a dictionary of ingredients with emojis
    """
    return [{'name': ingredient, 'emoji': INGREDIENTS_DICT.get(ingredient, "🍽️")} for ingredient in ingredients]

if __name__ == '__main__':
    app.run(port=8000, debug=True)