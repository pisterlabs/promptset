import openai
import time
import configparser
import logging
import json

# Configuration and Constants
MAX_RETRIES = 1 # Maximum number of retries for OpenAI API calls
_prompt_cache = {} # Cache for prompt templates

def load_openai_api_key(config_path='config.ini'):
    """
    Load the OpenAI API key from a configuration file.

    Parameters:
    - config_path (str): Path to the configuration file. Default is 'config.ini'.

    Returns:
    - str: OpenAI API key.

    Raises:
    - ValueError: If the API key is not found in the config file.
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    
    try:
        api_key = config['openai']['api_key']
    except KeyError:
        raise ValueError("Error: Could not retrieve the API key from config.ini. Make sure it's set correctly under [openai] section.")
    
    return api_key

# Set the OpenAI API Key
openai.api_key = load_openai_api_key()

def generate_prompt(job, parameters=None):
    """
    Generate a prompt based on a template and optional parameters.

    Parameters:
    - job (str): The key identifying which prompt template to load.
    - parameters (list): Optional list of parameters to format the template with.

    Returns:
    - str: Formatted prompt string or an empty string if prompt file not found.
    """
    if job in _prompt_cache:
        template = _prompt_cache[job]
    else:
        try:
            with open("prompts/" + job + ".txt", mode='r') as file:
                template = file.read()
            _prompt_cache[job] = template
        except FileNotFoundError:
            logging.error(f"Error: 'prompts/{job}.txt' not found!")
            return ""
    
    if parameters is None:
        return template
    
    return template.format(*parameters)

def call_openai(prompt, max_tokens=256, temp=0.7):
    """
    Call the OpenAI API with a given prompt and return the response.

    Parameters:
    - prompt (str): The text prompt to provide to the OpenAI API.
    - max_tokens (int): The maximum length of the response in terms of tokens. Default is 256 tokens.
    - temp (float): Temperature setting to control randomness of output. Default is 0.7.

    Returns:
    - str: The response from the OpenAI API or an empty string if maximum retries are reached.
    """
    response = None
    retry_count = 0  # Initialize retry counter
    retry_delay = 0.5  # Initial retry delay in seconds

    while response is None and retry_count < MAX_RETRIES:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": prompt}
                ]
            )
            response = completion["choices"][0].message.content.strip()
            # if not is_valid_json(response):
            #     logging.error("Error: The response is not a valid JSON format.")
            #     response = None
        except Exception as err:
            logging.error(f"Error: {err}")
    
        if response is None:
            time.sleep(retry_delay)
            retry_count += 1 
            retry_delay *= 2 
    if response is None:
        logging.error("Error: Max retries reached. Unable to get response.")
        return '{"error": "Unable to get a valid response."}'
    return response

def is_valid_json(s):
    """
    Check if a given string is a valid JSON.

    Parameters:
    - s (str): The string to check.

    Returns:
    - bool: True if valid JSON, False otherwise.
    """
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False
    

def handle_json(response_string):
    """
    Handle the OpenAI response string to make it a valid JSON string.

    :param response_string: The raw OpenAI response string
    :return: The parsed JSON as a Python dictionary
    """
    response_string = response_string.replace("\n", "\\n")
    response_string = response_string.replace('": "', '": \\"').replace('...",', '\\"...,')
    return json.loads(response_string)
