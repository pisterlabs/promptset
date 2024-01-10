"""
This script is designed to evaluate the differences in response quality and content between GPT-3.5 and GPT-4.
The test is run using the testing_chatbot_gpt3_gpt3.py chatbot,
located at '\testing\testing_chatbot\testing_chatbot_gpt3_gpt3.py'. It is a copy of the latest intent-less chatbot
found at '\intent-less_chatbot\chatbot.py', with the key difference being the exclusion of human handoff functions
in the test chatbot and the additional 'model' parameter.

To run this script update the 'path' variable to the root directory of this project and add your OpenAI API key to
'openaiapikey.txt' in the root directory of this project.
"""


# 1. Set up-------------------------------------------------------------------------------------------------------------
# Define path to root directory and OpenAI API key
import sys
path = r'C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis' # Change
testing_path = path + r'\testing'
sys.path.append(testing_path)

from testing_chatbot.testing_functions import open_file
import openai
import os
# Set the API key as an environment variable
os.environ['OPENAI_API_KEY'] = open_file(path + '\openaiapikey.txt') # Add OpenAI API kex to this txt file
openai.api_key = os.getenv('OPENAI_API_KEY')

# Standard Libraries
import pandas as pd
import sys

# Libraries to run the chatbot
import importlib
from time import sleep


# 2. Run chatbot--------------------------------------------------------------------------------------------------------

# Import testing chatbot
sys.path.append(testing_path + r'\testing_chatbot')
module = importlib.import_module('testing_chatbot_gpt3_gpt4')

# Import questions
questions_df = pd.read_csv(testing_path + r'\testing_data\test_dataset.csv')
questions = questions_df['Beschreibung'].tolist()

# Initialize lists to store generated answers
l_generated_response_gpt3 = []
l_generated_response_gpt4 = []

# Text generation models that are tested
models = ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview']

for model in models:
    for question in questions:
        response = module.main(question, model)

        if model == 'gpt-3.5-turbo-1106':
            l_generated_response_gpt3.append(response)
        elif model == 'gpt-4-1106-preview':
            l_generated_response_gpt4.append(response)

        sleep(5)

# 3. Save generated answers to csv file---------------------------------------------------------------------------------

questions_df['generated_answers_gpt3'] = l_generated_response_gpt3
questions_df['generated_answers_gpt4'] = l_generated_response_gpt4

# questions_df.to_csv(testing_path + r'\testing_results\gpt3_and_gpt4_comparison.csv') # Uncomment to resave
