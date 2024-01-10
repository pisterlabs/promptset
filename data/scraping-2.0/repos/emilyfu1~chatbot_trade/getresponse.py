import openai
import pandas as pd
from dotenv import dotenv_values, find_dotenv
import os
import os.path as path
import json
import requests

def getresponse(query):

    # set key parameter(s)
    # this looks for your configuration file and then reads it as a dictionary
    config = dotenv_values(find_dotenv())
    key = config["KEY"]
    base = config["ENDPOINT"]
    openai.api_type = "azure"
    openai.api_base = base
    openai.api_version = "2023-07-01-preview"
    openai.api_key = key

    response = openai.ChatCompletion.create(
    engine="gpt3turboDAZ",
    messages = [{"role":"system","content":query}],
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)

    return response.choices[0].message['content']