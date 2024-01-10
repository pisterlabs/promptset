#-*- coding:utf-8 -*-

import openai
import os
import pathlib

key_file = pathlib.Path.cwd() / 'openai_api_key.txt' # \CodingTest\back
openai.api_key = key_file.read_text(encoding='utf-8')

comment = "\n\n\"\"\"\nHere's what the above function is doing:\n 1."

def explain_code(user_code):
    response = openai.Completion.create(
        model="text-curie-001",
        prompt=user_code + comment,
        temperature=0.7,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\"\"\""]
    )
    result = "Here's what the above function is doing: \n 1." + response.choices[0].text
    return result

""" 
result example
Here's what the above function is doing: 
 1. It creates two variables, _curr and _next.
 2. It sets _curr to 0 and _next to 1.
 3. It loops through the number 3, counting down from 3 and adding 1 to _curr each time.
 4. When it gets to 1, it returns _curr.
"""