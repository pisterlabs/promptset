#!/usr/bin/env python3

import json
import yaml
from yaml import CLoader, CDumper
from revChatGPT.V3 import Chatbot
import os
from sys import argv
import fmtutil
from parse import py
from argparse import ArgumentParser
import openai

file_path = os.path.dirname(os.path.realpath(__file__))
with open(f"{file_path}/config.yaml", "r") as file:
    config = yaml.load(file, Loader=CLoader)

# Set the OpenAI API key using the value from the 'OPENAI_API_KEY' key in the config file
openai.api_key = config['OPENAI_API_KEY']
messages = [ {"role": "system", "content": 
              "You are a intelligent assistant."} ]
code = py.read_file_contents('example.py')

# Open the 'example-doc.md' file for writing documentation
with open(f"example-doc.md", "w") as outfile:
	outfile.write(f"# Documentation for `{argv[1]}`\n\n")
	head_ask = "Generate python docstrings for the given modules and functions. Add the documentations and code together:" + code
	messages.append(
		{"role": "user", "content": head_ask},
	)
    # Create a chat conversation with OpenAI's GPT-3.5 model
	chat = openai.ChatCompletion.create(
		model="gpt-3.5-turbo", messages=messages
	)
    # Get the response from the chat model
	resp = chat.choices[0].message.content
	print(resp)
	print(f'Generated documentation for example file.')
	output = f"### example\n" + fmtutil.highlight_multiline_code_md(resp, "python") + "\n\n"
	outfile.write(output)
