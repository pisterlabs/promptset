import base64
from google.cloud import vision
import requests
import json
import openai
import os
from flask_cors import CORS, cross_origin

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/arjun/Downloads/letmecook-83c018333242.json"

from flask import Flask, request, jsonify # Imports the flask library modules
app = Flask(__name__)
app.config['CORS_HEADERS']='Content-Type'
CORS(app, expose_headers='Authorization')

@app.route('/scanner', methods=['POST'])
def detect_objects():
    print(request.get_json())
    data = request.get_json()

    # Extract data from the JSON request
    meal = data.get('meal')
    restrictions = data.get('restrictions')
    allergies = data.get('allergies')
    calories = data.get('calories')
    method = data.get('method')
    time = data.get('time')
    base64_image = data.get('image')

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=base64.b64decode(base64_image[23:]))
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations
    print(objects)

    detected_objects = []
    for obj in objects:
        detected_objects.append(obj.name)
    detected_objects.append("NA")

    #get_nutritional_info(detected_objects)
    response = getRecipe(meal, restrictions, allergies, calories, method, time, detected_objects)
    print(response)
    return str(response)


def get_nutritional_info(ingredients):

        ingredients = ingredients.split(',')
        
        for food in ingredients:
            # Send a GET request
            response = requests.get("https://api.edamam.com/api/food-database/v2/parser?app_id=ff34d99d&app_key=b11c1fafb92403097c9bbdb5b8dcde7d&ingr=" + food + "&nutrition-type=cooking")

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Extract JSON data from the response content
                data = response.json()

                if 'parsed' in data and data['parsed']:
                    ingredients.append(food)
                    food_data = data['parsed'][0]['food']

                    nutrients = food_data.get('nutrients', {})

                    calories = nutrients.get('ENERC_KCAL', 'N/A')
                    protein = nutrients.get('PROCNT', 'N/A')
                    fat = nutrients.get('FAT', 'N/A')
                    carbs = nutrients.get('CHOCDF', 'N/A')
                    fibers = nutrients.get('FIBTG', 'N/A')

                    print("Calories: ", calories)
                    print("Protein: ", protein)
                    print("Fat: ", fat)
                    print("Carbohydrates: ", carbs)
                    print("Fibers: ", fibers)
                else:
                    print("Food data not found in the response.")
            else:
                print("Failed to fetch data. Status code:", response.status_code)

def getRecipe(mealVal, restrictions, allergiesVal, calories, method, time, ingredientsVal):
    #-------------------------------------------------------------------- 
    meal = mealVal

    dietaryRestrictions = restrictions

    allergies = allergiesVal

    cals = calories

    cookingMethods = method

    timeLimit = time

    ingredients = ingredientsVal

    api_key = "HIDDEN"
    openai.api_key = api_key

    prompt = f"Create a new recipe using only these ingredients (you don't need to use all of them but you cannot use something that I am not giving you): {', '.join(ingredients)}. The recipe should include detailed instructions. Here are the 'dietary_restrictions': {', '.join(dietaryRestrictions)}. Here is what the user is allergic to: {', '.join(allergies)}. Here are possible ways the user can cook the meal: {', '.join(cookingMethods)}. The max time limit for the meal is {timeLimit}. Here is the MAX number of calories the meal should be: {cals}... Give me my response as a JSON with the 'name', 'ingredients', 'instructions' (unnumbered), 'nutritional_facts' as the keys. Under each ingredient include the 'portion', 'calories', 'protein', 'fat', 'carbs', and 'fibers'. Generate an array called 'dietary_restrictions' of all the dietary restrictions that the recipe follows"+'You should return an object in this format: { "name": "<name of the dish>", "ingredients": [ { "ingredient": "<Name of the ingredient>", "portion": "<Portion or measurement>", "calories": <Calories count>, "protein": <Protein count>, "fat": <Fat count>, "carbs": <Carbohydrate count>, "fibers": <Fiber count> }, ... // Repeated sections for other ingredients ], "instructions": [ "<Step 1 of cooking (unnumbered)>", "<Step 2 of cooking (unnumbered)>", ... // Repeated steps for all the cooking instructions (unnumbered)], "nutritional_facts": { "calories": <Total calories>, "protein": <Total protein>, "fat": <Total fat>, "carbs": <Total carbs>, "fibers": <Total fibers> }, "dietary_restrictions": ["<Dietary restriction 1>", "<Dietary restriction 2>", ...] }'

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates recipes."},
            {"role": "user", "content": prompt}
        ]
    )

    recipe = response['choices'][0]['message']['content'].strip()

    return recipe

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=False, host="127.0.0.1", port=5505)