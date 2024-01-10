#!/usr/bin/python3
import os
import sys
import openai
import re
from dotenv import load_dotenv

from train import read_file

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

if len(sys.argv) < 2:
  print("Usage: python3 main.py <command>")
  sys.exit(1)

def command_to_code(command):
  command = command.lower().strip()
  training_code = read_file("training.js")
  # First check if the code is not already in the training set
  command_regex = re.escape(command)
  code_regex = '^ {4}case "'+command_regex+'":\\n( {4}case ".+?":\\n)*([\s\S]+?)\n {6}break;'
  code_search = re.search(code_regex, training_code, re.MULTILINE)
  if code_search:
    prediction = code_search.group(2)
    return unindent(prediction)
  
  # If not, ask OpenAI to generate a code
  command_token = "<command>"
  placeholder_token = "/* placeholder */"
  break_code = "\n      break;"
  code_start_index = training_code.find(placeholder_token)
  code_start = training_code[0:code_start_index]
  code_start = code_start.replace(command_token, command)

  response = openai.Completion.create(
    engine="davinci-codex",
    prompt=code_start,
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=[break_code]
  )
  prediction = response["choices"][0]["text"]
  return unindent(prediction)

def unindent(code):
  return re.sub(r'^ {6}', '', code, flags=re.MULTILINE)
print(command_to_code(sys.argv[1]))
