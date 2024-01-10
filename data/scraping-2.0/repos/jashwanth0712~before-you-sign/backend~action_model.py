import os
import openai
import json
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")



def action_function(legal_document_text):
  
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system","content":"Output all the key sentences that are really important in a legal,financial,fiduciary and regulatory  perspective as a proper JSON array, say or do nothing else, just output the list"},
              {"role": "user", "content":legal_document_text}],
    temperature = 0.5
  )
  #returns output as a list
  return json.loads(completion["choices"][0]["message"]["content"])

