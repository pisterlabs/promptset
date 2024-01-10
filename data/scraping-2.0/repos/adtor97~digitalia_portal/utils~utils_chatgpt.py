import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc
from dash import dash_table
from dash import html
from flask import render_template, json
from datetime import datetime, date
from utils import utils_google, utils
import openai
import requests
import os

api_key=os.environ["CHATGPT_API_KEY"]

def chat_chatgpt(messages, model="gpt-3.5-turbo", api_key=api_key):
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result
