from openai import OpenAI
import json

from backend.app.models.recipe_model import Recipe

openai_client = OpenAI()

def generate_recipe(ingredients) -> Recipe:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a helpful assistant which generates recipes given ingredients in JSON format. It should contain recipe_name, servings, ingredients and instructions. Make sure there is no nested json inside ingredients."},
                {"role": "user", "content": "Ingredients: " + ingredients},
            ]
        )
        recipe_json = json.loads(response.choices[0].message.content) # type: ignore
        recipe = Recipe(recipe_name=recipe_json['recipe_name'], servings=recipe_json['servings'], ingredients=recipe_json['ingredients'], instructions=recipe_json['instructions'])
        return recipe
    except Exception as e:
        raise e