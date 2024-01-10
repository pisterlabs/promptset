import requests
import time
from openai import OpenAI
import requests
import os
from halo import Halo
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console

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
    requests.post(f'{api_url}/south_bound_bus_insert', json=data, headers={'Content-Type': 'application/json'})
    return True

# post the message to the southbound bus.
def post_northbound_bus_message(data):
    # post the message to the southbound bus.
    requests.post(f'{api_url}/north_bound_bus_insert', json=data, headers={'Content-Type': 'application/json'})
    return True


# function to go to the next layer.
def nxtLayer(nextLayer):

    # active layer varibale
    current_Layer = None
    url = None

    layers = [
        {
            'layer': 'aspirational',
            'url': 'http://127.0.0.1:6061/aspirational_layer'
        },
        {
            'layer': 'globalStrategy',
            'url': 'http://127.0.0.1:6062/global_strategy_layer'
        },
        {
            'layer': 'agentModel',
            'url': 'http://127.0.0.1:6063/agent_model_layer'
        },
        {
            'layer': 'executiveFunction',
            'url': 'http://127.0.0.1:6064/executive_functions'
        },
        {
            'layer': 'CongnitiveControl',
            'url': 'http://127.0.0.1:6065/congnitive_control_layer'
        },
        {
            'layer': 'TaskProsecution',
            'url': 'http://127.0.0.1:6066/task_prosecution'
        }
    ]

    # if passed later in the list, we set it as the current layer.
    for layer in layers:
        if layer['layer'].lower() == nextLayer.lower():
            
            # set variables.
            current_Layer = layer
            url = layer['url']

        else:
            pass

    # get southbound bus messages.
    s_msgs = get_southbound_bus_messages()

    # configure the data to send to next layer.
    data = {
        'data': s_msgs[:6]
    }

    
    try:
        # make the request.
        response = requests.post(url, json=data)
        return response
    except Exception as e:
        print(e)
        pass

    # print(response)
    return current_Layer

# test next layer
# print(nxtLayer('aspirational'))