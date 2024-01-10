import openai
import os
from dotenv import load_dotenv
import json
from flask import Blueprint, request, jsonify, session


load_dotenv()
openai.api_key = os.getenv('OPEN_AI_API_KEY')
model_engine = "gpt-3.5-turbo-1106"
promptFile = open('prompt.txt', 'r')
prompt = promptFile.read()

GPT_messages = [
    {"role": "system", "content": prompt}
]
GPT_functions = [
    {
        "name": "give_recipes",
        "description": "Give three recipes to the user, including ingredients and instructions.",
        "parameters": {
                "type": "object",
                "properties": {
                    "recipe_one_name": {
                        "type": "string","description": "The name of the recipe",},
                    "recipe_one_ingredients": {
                        "type": "string", "description": "the ingredients of the recipe"},
                    "recipe_one_instructions": {
                        "type": "string", "description": "step by step instructions for the recipe"},
                    "recipe_two_name": {
                        "type": "string","description": "The name of the recipe",},
                    "recipe_two_ingredients": {
                        "type": "string", "description": "the ingredients of the recipe"},
                    "recipe_two_instructions": {
                        "type": "string", "description": "step by step instructions for the recipe"},
                    "recipe_three_name": {
                        "type": "string","description": "The name of the recipe",},
                    "recipe_three_ingredients": {
                        "type": "string", "description": "the ingredients of the recipe"},
                    "recipe_three_instructions": {
                        "type": "string", "description": "step by step instructions for the recipe"},
                    },
                },
    },
]

def give_recipes(recipe_one_name, recipe_one_ingredients, recipe_one_instructions, recipe_two_name, recipe_two_ingredients, recipe_two_instructions, recipe_three_name, recipe_three_ingredients, recipe_three_instructions):
    recipes = []
    if recipe_one_name != None:
        recipes.append({
            'name': recipe_one_name,
            'ingredients': recipe_one_ingredients,
            'instructions': recipe_one_instructions
        })
    if recipe_two_name != None:
        recipes.append({
            'name': recipe_two_name,
            'ingredients': recipe_two_ingredients,
            'instructions': recipe_two_instructions
        })
    if recipe_three_name != None: 
        recipes.append({
            'name': recipe_three_name,
            'ingredients': recipe_three_ingredients,
            'instructions': recipe_three_instructions
        })
    return recipes

recipes_bp = Blueprint('recipes', __name__)
@recipes_bp.route('/get_recipes', methods=['POST'])
def get_recipes():
    data = request.get_json()
    ingredients = data.get('ingredients', [])  # Fetch 'ingredients' key if it exists, otherwise set to an empty list

    ingredients_string = ", ".join(ingredients)  # Convert list to a comma-separated string

    GPT_messages.append({"role": "user", "content": ingredients_string})

    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=GPT_messages,
        temperature=0.5,
        functions=GPT_functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )

    #GPT_response = response['choices'][0]['message']['content']
    GPT_response = response['choices'][0]['message']

    if 'function_call' in GPT_response:
        available_functions = {
            "give_recipes": give_recipes,
        }
        function_name = GPT_response["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(GPT_response["function_call"]["arguments"])
        function_response = function_to_call(
            recipe_one_name=function_args.get("recipe_one_name"),
            recipe_one_ingredients=function_args.get("recipe_one_ingredients"),
            recipe_one_instructions=function_args.get("recipe_one_instructions"),
            recipe_two_name=function_args.get("recipe_two_name"),
            recipe_two_ingredients=function_args.get("recipe_two_ingredients"),
            recipe_two_instructions=function_args.get("recipe_two_instructions"),
            recipe_three_name=function_args.get("recipe_three_name"),
            recipe_three_ingredients=function_args.get("recipe_three_ingredients"),
            recipe_three_instructions=function_args.get("recipe_three_instructions"),
        )
        session['recipes'] = function_response

        return function_response
    return GPT_response
#Example of calling the function
#get_recipies("bread, milk, sugar, salt, eggs, flour")
