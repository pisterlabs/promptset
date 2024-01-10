"""
This script runs the latest set-up of the intent-less chatbot on questions related to one of the intents that the
intent-based rasa chatbot was trained on. For each intent, ten questions are posed to the chatbot.
These same questions have also been answered by the intent-based chatbot, enabling a comparison of performance
between the two bots.

The script for this testing chatbot testing, located at '\testing\testing_chatbot\rasa_openai_test_chatbot.py',
mirrors the latest intent-less chatbot found at '\intent-less_chatbot\chatbot.py', with the key difference being
the exclusion of human handoff functions in the test chatbot

To run this script update the 'path' variable to the root directory of this project and add your OpenAI API key to
'openaiapikey.txt' in the root directory of this project.
"""


# 1. Set up-------------------------------------------------------------------------------------------------------------
# Define path to root directory and OpenAI API key
path = r'C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis'
testing_path = path + r'\testing'
import sys
sys.path.append(testing_path)

from testing_chatbot.testing_functions import open_file
import openai
import os
os.environ['OPENAI_API_KEY'] = open_file(path + '\openaiapikey.txt')
openai.api_key = os.getenv('OPENAI_API_KEY') # Add OpenAI API key to this .txt file

# Import libraries and functions
# Standard Libraries
import pandas as pd

# Libraries to run chatbot
from time import sleep

# Libraries to import chatbotL
import importlib
import sys


# Import testing chatbot
sys.path.append(path + r"\testing\testing_chatbot")
module = importlib.import_module("rasa_openai_test_chatbot")

# Import questions
questions_df = pd.read_csv(testing_path + r"\testing_data\rasa_openai_comparison.csv")
questions = questions_df["question"].tolist()


# 2. Run chatbot--------------------------------------------------------------------------------------------------------
l_generated_response = [] # Initialize list to store the generated responses
for question in questions:
    response = module.main(question)
    l_generated_response.append(response)
    sleep(30)

# 3. Save the generated responses in a csv file-------------------------------------------------------------------------
questions_df["generated_answers"] = l_generated_response
#questions_df.to_csv(testing_path + r"\testing_results\final_test.csv") # Uncomment to resave