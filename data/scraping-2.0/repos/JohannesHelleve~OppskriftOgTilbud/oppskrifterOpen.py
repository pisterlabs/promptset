import openai
import os
from dotenv import load_dotenv
from OppskriftOgTilbud import push_to_mongo
import json

load_dotenv()

OpenAiAPI = os.getenv('OpenAI')

openai.api_key = OpenAiAPI

def make_recipe():
  oppskrift = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"system", "content": "Lag en oppskrift , formater oppskriften som markdown\n\nInkluder navnet på matretten, en beskrivelse på ca 200 ord i begynnelsen av oppskriften og en liste over alle ingrediensene.\n\nEksempel på ingredientsliste format\n\n- 2 fedd [hvitløk]\n- Salt og pepper\n- 1 ts [paprikapulver]\n- 1 ts [tørket oregano]"}],
    temperature=1,
    max_tokens=1505,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  jsonOppskrift = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"system", "content": f"Extract the following information:\n- title\n- description\n- ingredients (array with keys amount, unit, text, keyword, include all of them, let the value be null if it can't be extracted),\n- steps (array of strings)\n\nRECEIPE\n\n{oppskrift}\n\nOutput in JSON:"}],
    temperature=1,
    max_tokens=1505,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  return jsonOppskrift

#kan brukes for å finne ut vekt av en del av ingrediensen, men og hente ut vekt fra title til ingrediensen. 
def get_weight_ingrdient(ingredient):
  vekt = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"system", "content": f"Hvor mange gram veier en {ingredient}?"}],
    temperature=1,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  vektDict = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"system", "content": f"Fra følgende tekst hent navn på ingrediensen og vekt i gram, og lagre svarene i json format. La vekten være Null hvis den ikke kan bli hentet. {vekt}"}],
    temperature=1,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  if isinstance(vektDict, dict):
    return vektDict
  else:
    raise Exception("Open didnt returned a dict, try the function again")

def get_ingredients(message):
    recipeJson = make_string_dict(message)
    ingredients = []
    i = 0
    recipeItemsLen = len(recipeJson['ingredients'])
    while i < recipeItemsLen:
        ingredient = [recipeJson["ingredients"][i]["amount"], recipeJson["ingredients"][i]["unit"], recipeJson["ingredients"][i]["keyword"]]
        ingredients.append(ingredient)
        i += 1
        if i == recipeItemsLen :
            break
    return ingredients

def make_string_dict(message):
    string = message['choices'][0]['message']['content']
    stringAsDict = json.loads(string)
    return stringAsDict

