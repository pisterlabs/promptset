# Access OpenAI Codex API
import argparse
import json
import os

import openai

# Retrieve API from json file
with open('/openai/.openai/api_key.json') as f:
    api = json.load(f)

# set API key
openai.api_key = api['key']

# Function to translate SAS code to Python
def translate_sas_to_python(prompt, sas_code):
    completion_prompt = f'{prompt}:\n\n{sas_code}\n\nPython code:'

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",   #'text-davinci-003',
        messages=[{"role": "user", "content": completion_prompt}],
        max_tokens=2048,
        temperature=0.0,
        n=1,
        stop=None,
    )

    python_code = response.choices[0].message.content.strip()
    return python_code

if __name__ == '__main__':
    # retrieve sas file name from first parameter in command line
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", help="Source file for conversion")

    # retrieve output directory from second parameter in command line
    parser.add_argument("target_file", help="Target file of conversion")

    #retrieve prompt from third parameter in command line, with default value
    parser.add_argument(
        "--prompt",
        help="Prompt to use for conversion",
        default="convert this SAS program to Python")

    args = parser.parse_args()

    # read in sas file
    with open(args.source_file, 'r') as f:
        source_code = f.read()

    print(f"starting conversion for {args.source_file}...with prompt '{args.prompt}'")
    # translate sas code to python
    prompt = args.prompt
    target_code = translate_sas_to_python(prompt, source_code)

    # write target file
    target_file = args.target_file
    with open(target_file, 'w') as f:
        f.write(target_code)

    print(f"finished conversion for {args.source_file}...created {target_file}")




