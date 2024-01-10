from mergedeep import Strategy, merge
import logging
import os
import json
import openai
from .models import ImproveAIResponse

from common import find_root_dir
from misc.dict_loader import get_dict_from_file
from ai.utils import get_api_key, get_prompt_file
from ai.prompt import write_prompt_topology_file
from misc.io_utils import JSON_YAML_FIND_PATTERN, find_file_with_pattern


# The prompt dictionary
PROMPT_DOCS_DIR = os.path.join(os.path.dirname(__file__), 'prompt_docs')

# Get the example topology file
EXAMPLE_TOPOLOGY_FILE = os.path.join(PROMPT_DOCS_DIR, 'ai_example.yaml')
EXAMPLE_TOPOLOGY = get_dict_from_file(EXAMPLE_TOPOLOGY_FILE)

# Get the schema file
SCHEMA_FILE = os.path.join(PROMPT_DOCS_DIR, 'ai_schema.json')
SCHEMA = get_dict_from_file(SCHEMA_FILE)


def generate_ai_topology():
    # Get the API key
    openai.api_key = get_api_key()

    # Get the prompt from the user
    user_input = input("What network topology you want to build? 🏗️️👷\n")

    context = {'example_topology': json.dumps(EXAMPLE_TOPOLOGY),
               'schema': json.dumps(SCHEMA),
               'user_prompt': user_input}

    # Parse the prompt file
    openai_prompt_file = get_prompt_file('generate_topology', context)

    messages = openai_prompt_file['messages']
    ai_params = openai_prompt_file['config']

    # Create the response
    response = openai.ChatCompletion.create(messages=messages, **ai_params)

    # Get the json string output
    json_str = response["choices"][0].message.content

    # Now load it into a file
    json_dict = json.loads(json_str)

    # Get the file name from the user
    file_name = input(
        f'Here is your topology 🚀!\n${json_str}\nWhat will be the name of the topology file (for example: my_topology.yaml) ')

    # If we got the file name
    if file_name:
        root_dir = find_root_dir()
        file_path = os.path.join(root_dir, 'topologies', file_name)

        # Write the topology file
        write_prompt_topology_file(json_dict, file_path)


def improve_ai_topology(topology_file_name: str):
    # Get the API key
    openai.api_key = get_api_key()

    # Get the root directroy
    root_dir = find_root_dir()

    # Get the topologies directory
    topologies_dir = os.path.join(root_dir, 'topologies')

    # The topology file path
    topology_file = find_file_with_pattern(
        topologies_dir, topology_file_name + JSON_YAML_FIND_PATTERN)

    if not topology_file:
        logging.error(f'The topology {topology_file_name} was not found!')
        exit(1)

    # Get the dictionary of this topology
    topology_dict = get_dict_from_file(topology_file)

    # Get the prompt from the user
    user_input = input(
        "What would you like to modify/add in this topology 🛠️ ? ")

    context = {'schema': json.dumps(SCHEMA),
               'user_prompt': user_input,
               'user_topology': topology_dict}

    # Parse the prompt file
    openai_prompt_file = get_prompt_file('improve_topology', context)

    messages = openai_prompt_file['messages']
    ai_params = openai_prompt_file['config']

    # Create the response
    response = openai.ChatCompletion.create(messages=messages, **ai_params)

    # Get the json string output
    ai_output = response["choices"][0].message.content

    # Now load it into a file
    ai_dict: ImproveAIResponse = json.loads(ai_output)

    # The new modified topology
    modified_topology = ai_dict['json']

    # # Now merge the dictionaries
    # merge(topology_dict, modified_topology,
    #       strategy=Strategy.REPLACE)

    # Get the file name from the user
    file_name = input(
        f'{ai_dict["message"]}\nWhat will be the name of the new modified topology file (for example: my_topology.yaml) ')

    # If we got the file name
    if file_name:
        file_path = os.path.join(root_dir, 'topologies', file_name)

        # Write the topology file
        write_prompt_topology_file(modified_topology, file_path)
