from flask import Flask
import requests
from flask import jsonify
from flask import Flask, abort, request
from tempfile import NamedTemporaryFile
import whisper
import torch
from pathlib import Path
import os
import json 
import gpt3
import openai

# GPT-3 API Key
openai.api_key = "" #TODO: instert the API

# Check if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Whisper model:
model = whisper.load_model("base", device=DEVICE)

app = Flask(__name__)


def recipe_from_natural(dish: str, tp="main course", diet="", exc="", intol=""):
    url = "https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/search"

    querystring = {"query": dish, "number": "5", "offset": "0", "type": tp}

    if diet != "":
        querystring["diet"] = diet

    if exc != "":
        querystring["excludeIngredients"] = exc

    if intol != "":
        querystring["intolerances"] = intol

    headers = {
        "X-RapidAPI-Key": "128c1f46a4msha595cbdd6293956p142d78jsnc0f4b19ff043",
        "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
    }

    response = json.loads(requests.request("GET", url, headers=headers, params=querystring).text)["results"]
    return response

def recipe_from_ingredents(ingredients: str):
    url = "https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/findByIngredients"

    querystring = {"ingredients": ingredients, "number": "5", "ignorePantry": "true", "ranking": "1"}

    headers = {
        "X-RapidAPI-Key": "128c1f46a4msha595cbdd6293956p142d78jsnc0f4b19ff043",
        "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
    }

    response = json.loads(requests.request("GET", url, headers=headers, params=querystring).text)

    return response

def info_from_recipe(recipe: dict):
    id = recipe["id"]

    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{id}/information"

    querystring = {"includeNutrition": "true"}

    headers = {
        "X-RapidAPI-Key": "128c1f46a4msha595cbdd6293956p142d78jsnc0f4b19ff043",
        "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
    }

    response = json.loads(requests.request("GET", url, headers=headers, params=querystring).text)["results"]

    return json.loads(response)


def place_from_dish(dish: str):
    url = "https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/menuItems/search"

    querystring = {"query": dish, "offset": "0", "number": "5", "minCalories": "0", "maxCalories": "5000",
                   "minProtein": "0", "maxProtein": "100", "minFat": "0", "maxFat": "100", "minCarbs": "0",
                   "maxCarbs": "100"}

    headers = {
        "X-RapidAPI-Key": "128c1f46a4msha595cbdd6293956p142d78jsnc0f4b19ff043",
        "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
    }

    response = json.loads(requests.request("GET", url, headers=headers, params=querystring).text)["menuItems"]

    return response

def info_from_placedish(pd):
    id = pd["id"]

    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/menuItems/{id}"

    headers = {
        "X-RapidAPI-Key": "128c1f46a4msha595cbdd6293956p142d78jsnc0f4b19ff043",
        "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers).text

    return response

def food_from_text(text):
    url = "https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/detect"
    text = text.replace(" ", "%20")
    payload = f"text={text}"

    headers = {

        "content-type": "application/x-www-form-urlencoded",
        "X-RapidAPI-Key": "128c1f46a4msha595cbdd6293956p142d78jsnc0f4b19ff043",
        "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
    }

    resp = ""
    response = json.loads(requests.request("POST", url, data=payload, headers=headers).text)["annotations"]
    for i in range(len(response)):
        if i != len(response) - 1:
            resp += response[i]["annotation"] + ", "
        else:
            resp += response[i]["annotation"]
    return resp

def gtp3_request(i):
    q1 = "Which is the main dish?"
    q2 = "What the customer dont want to eat?"
    q3 = "Is it a drink?"

    openai.api_key = "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"
    response = openai.Completion.create(
        model="text-curie-001",
        prompt=f"Read this customer response then answer the following questions:\n\n\"\"\"\n{i}\n\"\"\"\n\n"
               f"Questions:\n1. {q1}\n2. {q2}\n3. {q3}\n\nAnswers:\n1.",
        temperature=0.1,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0.37,
        presence_penalty=0,
        stop=["\n\n"]
    )
    print(response)
    return response["choices"][0]["text"].split("\n")


def gpt3_to_recipe(from_whisper: str):
    response = gtp3_request(from_whisper)
    if "no" in response[-1]:
        tp = "main course"
    else:
        tp = "drink"
    exc = food_from_text(response[1])
    val = response[0].replace("The main dish is a", "")
    try:
        val = val.split(" without")
    except:
        pass
    resp = recipe_from_natural(val[0], exc=exc)
    if not resp:
        resp = "No recipes were found as requested"

    return resp

def gpt3_to_store(from_whisper: str):
    response = gtp3_request(from_whisper)
    val = response[0].replace("The main dish is a", "")
    try:
        val = val.replace(".", "")
    except:
        pass
    resp = place_from_dish(val)
    if not resp:
        resp = "No dishes were found for purchase as requested"
    return resp

@app.route("/")
def hello_world():
    print("HELLOW WORLD!")
    rta = {'greeting':'Hello from Flask!'}
    return jsonify(rta)

@app.route('/processAudio',methods=['POST'])
def processAudio():
    print("Llego")
    return "<h1> Super bien </h1>"

@app.route('/whisper', methods=['POST'])
def handler():
    body = request.get_json()
    if len(body)==0:
        # If the user didn't submit any files, return a 400 (Bad Request) error.
        abort(400)
    body=json.loads(body["text"])
    transcript = body["text"]
    summary = gpt3.gpt3complete(transcript).replace("\n", "")
    stores = gpt3_to_store(transcript)
    recipes =  gpt3_to_recipe(transcript)
    return {'transcript': transcript, 
            "summary":summary, 
            'recipes': recipes, 
            'stores': stores}

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
