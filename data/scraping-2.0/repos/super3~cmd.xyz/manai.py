#!/usr/bin/env python
import sys
import openai
from config import *
openai.api_key = api_key

prompt = """
Q: list files
> ls

Q: list files including hidden files
> ls -a

Q: list files and sort files by size
> ls -lS

Q: list files in reverse order
> ls -r

"""

from flask import Flask, escape, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    usr_prompt = prompt + "Q: " + request.args.get("input", "")
    response = openai.Completion.create(engine="davinci", prompt=usr_prompt, temperature=0, stop="\n\n")
    print(usr_prompt, response.choices[0].text)

    res = jsonify({"output": response.choices[0].text})
    res.headers.add('Access-Control-Allow-Origin', '*')

    return res
