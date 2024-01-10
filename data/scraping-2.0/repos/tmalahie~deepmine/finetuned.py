#!/usr/bin/python3
import os
import sys
import openai
import re
import json
from dotenv import load_dotenv

from train import code_prefix, code_suffix, read_file

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

if len(sys.argv) < 2:
  print("Usage: python3 main.py <command>")
  sys.exit(1)

def remove_code_bounds(code):
  prefix = code_prefix
  suffix = code_suffix
  return code[len(prefix):-len(suffix)]

def command_to_code(command):
  command = command.lower().strip()
  training_sample = read_file("training.jsonl")
  # First check if the code is not already in the training set
  for line in training_sample.split("\n"):
    if line:
      line_json = json.loads(line)
      if line_json["prompt"] == command:
        return remove_code_bounds(line_json["completion"])
  
  # If not, ask OpenAI to generate a code
  model = json.loads(read_file("model.json"))
  model_payload = openai.FineTune.retrieve(id=model["id"])
  if not model_payload["fine_tuned_model"]:
    raise Exception("Model is still training")
  
  response = openai.Completion.create(
    model=model_payload["fine_tuned_model"],
    prompt=command,
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=[code_suffix[1:]]
  )
  prediction = response["choices"][0]["text"]
  return prediction[len(code_prefix):]

print(command_to_code(sys.argv[1]))
