# activate env with: source env/bin/activate

import json
import os
from itertools import combinations
from dotenv import load_dotenv
import openai

# load translators.json
with open('translators.json') as translator_data_file, open('comparisons.json') as comparison_data_file:
  translator_data = json.load(translator_data_file)
  comparison_data = json.load(comparison_data_file)

# Create a list of unique translator pairs
translators = translator_data.keys()
unique_pairs = list(combinations(translators, 2))

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

comparsions_json = {}

for entry in unique_pairs:
  comparison_key = f"{entry[0]}-vs-{entry[1]}"
  alt_comparison_key = f"{entry[1]}-vs-{entry[0]}"

  # Skip if comparison already exists
  if comparison_key in comparison_data or alt_comparison_key in comparison_data:
    print(f'Comparison for {comparison_key} already exists. Skipping...')
    comparsions_json[comparison_key] = comparison_data[comparison_key]
    continue
  
  # Get the translator name and quote for each entry
  entry_0_name = translator_data[entry[0]]["translator"]
  entry_1_name = translator_data[entry[1]]["translator"]
  entry_0_quote = next(iter(translator_data[entry[0]]["quotes"].items()))[1]
  entry_1_quote = next(iter(translator_data[entry[1]]["quotes"].items()))[1]

  if not entry_0_quote or not entry_1_quote or not entry_0_name or not entry_1_name:
    print(f'Could not find quote or name for {entry[0]} or {entry[1]}. Skipping...')
    continue
  
  # Generate comparison
  print(f'Generating comparison for {comparison_key}...')

  prompt = f"Write a one or two sentence summary comparing the Iliad translations of {entry[0].capitalize()} and {entry[1].capitalize()}."

  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are an expert scholar on the Iliad by Homer."},
      {"role": "user", "content": prompt}
    ]
  )

  response_content = response["choices"][0]["message"]["content"]
  
  comparsions_json[comparison_key] = {
    "name": f"{entry[0].capitalize()} vs {entry[1].capitalize()}",
    "description": response_content
  }

# Create a new JSON file with the unique comparisons of translators
with open('new_comparisons.json', 'w') as outfile:
  json.dump(comparsions_json, outfile)


# response = {
#   "id": "chatcmpl-7V860EgjdVQnUzrve1pmdiRJpRNvR",
#   "object": "chat.completion",
#   "created": 1687653644,
#   "model": "gpt-3.5-turbo-0301",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "Robert Fagles' translation of the Iliad is a poetic masterpiece that captures the timeless tale of Achilles' wrath and the Trojan War with stunning clarity and vivid imagery."
#       },
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 24,
#     "completion_tokens": 35,
#     "total_tokens": 59
#   }
# }