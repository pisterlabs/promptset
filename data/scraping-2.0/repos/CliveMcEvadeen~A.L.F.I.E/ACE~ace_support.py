import requests
import json
import re 
import time
from openai import OpenAI
import requests
import os
from halo import Halo
from datetime import datetime
import textwrap
from dotenv import load_dotenv
from rich.console import Console
import google.generativeai as palm

# load .env file
load_dotenv()

# openai api key
key = os.getenv('open_ai_api_key')

# authenticate openai
client = OpenAI(api_key=key)

# define api url
api_url = 'http://192.168.1.5:6070'

# define console
console = Console()

# support functions.
@Halo(text='Congitive Processing...', spinner='dots')
def long_running_function(loading_message):
    spinner = Halo(text=f'{loading_message}', spinner='dots')
    spinner.start()
    time.sleep(0.5)
    spinner.stop()

# function to get the southbound bus messges.
def get_southbound_bus_messages():
    # get the bus messages.
    bus_messages = requests.get(f'{api_url}/south_bound_bus_get').json()
    return bus_messages


# get northbound bus messages.
def get_northbound_bus_messages():
    # get the bus messages.
    bus_messages = requests.get(f'{api_url}/north_bound_bus_get').json()
    return bus_messages


# post the message to the southbound bus.
def post_southbound_bus_message(data):
    # post the message to the southbound bus.
    requests.post(f'{api_url}/north_bound_bus_insert', json=data, headers={'Content-Type': 'application/json'})


# post the message to the southbound bus.
def post_northbound_bus_message(data):
    # post the message to the southbound bus.
    requests.post(f'{api_url}/north_bound_bus_insert', json=data, headers={'Content-Type': 'application/json'})


# test posting data to the bus.

data = {
    "layer": "aspirational-Test-layer",
    "messages": "This is a test message"
}

post_northbound_bus_message(data)

print(get_northbound_bus_messages())