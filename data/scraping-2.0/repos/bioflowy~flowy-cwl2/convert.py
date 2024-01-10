#!/bin/env python3

import os
import sys
import openai
import ast
# APIキーの設定
openai.api_key = ""
def convert_to_typescript(python_code):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"""Please convert the following piece of python code to typescript.
don't define missing types or functions.
output typescript code only.:\n\n{python_code}\n"""},
        ],
    )
    print(response.choices[0]["message"]["content"].strip())

python_code = ""
with open(sys.argv[1]) as f:
    python_code = ""
    for line in f:
        if line.startswith("#------"):
            convert_to_typescript(python_code)
            python_code = ""
        else:
            python_code = python_code + line
    convert_to_typescript(python_code)
