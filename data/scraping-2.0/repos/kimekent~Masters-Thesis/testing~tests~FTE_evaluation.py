"""
In this script the latest version of the intent-less chatbot answers web support questions.
The generated answers are manually checked to see if they are correct. This test is to see
how many questions the bot can correctly answer and consequently how large the arbeotsentlastugng is.

To run this script update the 'path' variable to the root directory of this project and add your OpenAI API key to
'openaiapikey.txt' in the root directory of this project.
"""

# 1. Set up-------------------------------------------------------------------------------------------------------------
# Set path to root directory and OpenAI API key
import sys
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis" # Change
testing_path = path + r'\testing'
sys.path.append(testing_path)

from testing_chatbot.testing_functions import open_file
import openai
import os
os.environ['OPENAI_API_KEY'] = open_file(path + '\openaiapikey.txt')
openai.api_key = os.getenv('OPENAI_API_KEY') # Add OpenAI API key to this .txt file

# Import packages
# Standard libraries
import pandas as pd

# Libraries to run chatbot
from time import sleep

# Libraries to import chatbotL
import importlib

# Import testing chatbot
sys.path.append(testing_path + r"\testing_chatbot")
module = importlib.import_module("rasa_openai_test_chatbot")

# Import questions
questions_df = pd.read_csv(testing_path + r"\testing_data\FTE_evaluation.csv")
questions = questions_df["Beschreibung"].tolist()
len(questions)

# 2. Run chatbot--------------------------------------------------------------------------------------------------------
l_generated_response = []  # Initialize list to store the generated responses
for question in questions:
    response = module.main(question)
    l_generated_response.append(response)
    sleep(30)

# 3. Save the generated responses in a csv file-------------------------------------------------------------------------
questions_df["generated_answers"] = l_generated_response
questions_df.to_csv(testing_path + r"\testing_results\fte_evaluation.csv")
