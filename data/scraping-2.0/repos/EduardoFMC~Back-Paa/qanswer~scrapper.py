import os
import csv
import json
import openai
from dotenv import load_dotenv

load_dotenv()

filedir = os.path.dirname(os.path.realpath(__file__))
openai.api_key = os.getenv("OPENAI_API_KEY")

class GptScrapper:
  def __init__(self, model_name = "gpt-3.5-turbo"):
    system_def_content = open(f'{filedir}/data/system.txt', mode="r", encoding="utf-8").read()
    self.system_def = { "role": "system", "content": system_def_content }

    self.model_name = model_name
    self.messages = []
    self.messages.append(self.system_def)
    self.total_tokens = 0

  def chat(self, message):
    self.messages.append({"role": "user", "content": message})

    response = openai.ChatCompletion.create(model=self.model_name, messages=self.messages)
    reply = response["choices"][0]["message"]["content"]
    self.total_tokens += response["usage"]["total_tokens"]
    
    self.messages.append({"role": "assistant", "content": reply})

    return reply
  
  def chat_without_context(self, message):
    internal_messages = [self.system_def, {"role": "user", "content": message}]
    response = openai.ChatCompletion.create(model=self.model_name, messages=internal_messages)
    reply = response["choices"][0]["message"]["content"]
    self.total_tokens += response["usage"]["total_tokens"]

    return reply

gptScrapper = GptScrapper()

while True:
  message = input('\nuser > ')
  reply = gptScrapper.chat_without_context(message)
  print(f'\n{gptScrapper.model_name} ({gptScrapper.total_tokens}) > {reply}')


# filedir = os.path.dirname(os.path.realpath(__file__))
# csv_file = open(f'{filedir}/data/1st_gen.csv', mode="r", encoding="utf-8")
# pokemons = csv.DictReader(csv_file)
# pokemon_names = [pokemon["name"] for pokemon in pokemons]

# intents = [] # { "tag": pokemons_name, "patterns": [], "responses": [pokemons_name] }
# filename = 'pguess_intents.json'
# errors = []

# for i, pokemon_name in enumerate(pokemon_names):
#   print(f'{i+1}/{len(pokemon_names)}: {pokemon_name}')
#   reply = ''
#   patterns = ''

#   try:
#     reply = gptScrapper.chat_without_context(pokemon_name)
#     patterns = reply
#   except:
#     print(f'Error: {pokemon_name}\n{reply}')
#     errors.append(pokemon_name)

#   intents.append({
#     "tag": pokemon_name,
#     "patterns": patterns,
#     "responses": [pokemon_name]
#   })
#   print(f'Total tokens: {gptScrapper.total_tokens}')

# with open(f'{filedir}/data/{filename}', mode="w", encoding="utf-8") as intents_file:
#   json.dump(intents, intents_file, indent=2)

# print(f"Added {len(intents)} questions to {filename}")
# print(f"Errors: {errors}")
