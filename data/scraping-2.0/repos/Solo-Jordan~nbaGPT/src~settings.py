"""
This file checks for the existence of a .env file and loads the environment variables from it. If the .env
file does not exist then we load the environment variables from the system environment variables.
"""

import os
from dotenv import load_dotenv
import logging
import openai


env_file = '../.env'
if os.path.exists(env_file):
    load_dotenv(dotenv_path=env_file)

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

# Suppress openai logging
if logging.getLogger('httpx').isEnabledFor(logging.DEBUG):
    logging.getLogger('httpx').setLevel(logging.INFO)
else:
    logging.getLogger('httpx').setLevel(logging.WARNING)

# Load the environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY
