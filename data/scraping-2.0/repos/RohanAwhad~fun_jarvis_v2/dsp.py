import g4f
from typing import Dict
import re
import json

import prompts

g4f.debug.logging = True  # Enable debug logging
g4f.debug.check_version = False

# def call_llm(prompt: str) -> str:
#   response = g4f.ChatCompletion.create(
#     model = g4f.models.gpt_4,
#     provider=g4f.Provider.Bing,
#     messages=[{"role": "user", "content": prompt}],
#     stream=False,
#   )
#   return response

from openai import OpenAI
client = OpenAI()

def call_llm(prompt: str, is_json_output: bool=True) -> str:
  if is_json_output:
    response = client.chat.completions.create(
      model="gpt-3.5-turbo-1106",
      response_format={'type': 'json_object'},
      messages=[{"role": "user", "content": prompt}],
      temperature=0.3,
      max_tokens=2000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
  else:
    response = client.chat.completions.create(
      model="gpt-3.5-turbo-16k",
      messages=[{"role": "user", "content": prompt}],
      temperature=0.3,
      max_tokens=4000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
  return response.choices[0].message.content

def get_json(text: str) -> Dict:
  json_string_extracted = re.search(r"\{.*\}", text, re.DOTALL).group()
  return json.loads(json_string_extracted)

def main(question: str) -> str:
  # first iteration
  first_call = prompts.initial_prompt.format(query=question)
  subquestion = get_json(call_llm(first_call))['factoid_subquestion']
  # search for answer for this subquestion
  #   embed
  #   retrieve
  #   rerank
  #   generate

  # continue with second iteration
  


if __name__ == "__main__":
  query = "Analyze the use of R-trees in mapping applications like Google Maps. Discuss the advantages of R-trees in spatial data handling and the challenges they might face in such applications"