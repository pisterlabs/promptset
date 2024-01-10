import os
import openai
import glob
import json
import util
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


openai.api_key = os.getenv("OPENAI_API_KEY")

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))



page_order = json.load(open('group-ui-util/anthony-correspondence-id-sets.json'))

for resource in glob.glob('anthony-correspondence-resources/*.json'):
  print(resource)
  
  data = json.load(open(resource))

  if 'gpt' not in data:
    data['gpt'] = {}

  if 'correspondence-summarize-4-sentences' in data['gpt']:
    continue

  full_text = data['full_text']

  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Using only the text below extract in full dictonary JSON format who this letter was sent to, who it was sent from, on what date in the format yyyy-mm-dd, summarize the contents in four sentences, and extract the names of the people mentioned using the dictonary keys recipient, sender, date, contents, peopleMentioned:\n---\n{full_text}\n---\n",
    temperature=0.25,
    max_tokens=506,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  data['gpt']['correspondence-summarize-4-sentences'] = response['choices'][0]
  
  
  json.dump(data,open(resource,'w'),indent=2)

