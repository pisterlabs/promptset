#!/usr/bin/python
import os
import argparse
import sys
from openaiConnector import *

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", help="Source file to be read", required=True)
parser.add_argument("-l", "--language", help="Language of the source file", default="python")
parser.add_argument("-a", "--allfile", help="Whether to generate code for the whole file or just for the code snippets", action="store_true")
args = parser.parse_args()

source = args.source
language = args.language
allfile = args.allfile

# Remove code blocks
def remove_code_blocks(code):
    new_code = ""
    for line in code.splitlines():
        if not line.startswith("```"):
            new_code += line + "\n"
    return new_code

# Load OpenAI API key
api_key = os.environ.get('OPENAI_API_KEY')
if api_key is not None:
    print("Loaded OpenAI API key from Environment", file=sys.stderr)
else:
    with open('api-key.txt') as f:
        api_key = f.read().strip()
    print("Loaded OpenAI API key from file.", file=sys.stderr)

# Check if API key is valid
check_api_key_validity(api_key)

# Open source file
with open(source) as s:
    if allfile:
        # Read whole file
        l = s.read()
        # Prompt
        prompt = "Write the raw valid {} code for this, ready to be executed, please include comments on each function:\n\n{}".format(language, l)
        # Get code from GPT
        code = call_AI_chatGPT(prompt)
        # Remove code blocks
        code = remove_code_blocks(code)
        # Print code
        print(code)
    else:
        keywords=['assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'lambda', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
        for line in s:
            if line.startswith(tuple(keywords)) or line.startswith("#"):
                print(line)
            else:
                # Prompt
                prompt = "Write the raw valid {} code for this, ready to be embedded into another {} code file:\n{}".format(language, language, line)
                # Get code from GPT
                code = call_AI_chatGPT(prompt)
                # Remove code blocks
                code = remove_code_blocks(code)
                # Print code
                print(code)
